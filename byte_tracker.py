# Ultralytics YOLO 🚀, GPL-3.0 license

import numpy as np

from ..utils import matching
from ..utils.kalman_filter import KalmanFilterXYAH
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, tlwh, score, cls):

        # wait activate
        self._tlwh = np.asarray(self.tlbr_to_tlwh(tlwh[:-1]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = tlwh[-1]

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class BYTETracker:

    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.trackslist = []

        self.frame_id = 0
        self.trackingid = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()

    def update(self, results, img=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        detwithid = []
        scores = results.conf
        bboxes = results.xyxy
        
        # add index
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        #print("bboxes = ", bboxes)
        cls = results.cls

        remain_inds = scores > self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]
        print("det = ", dets)
        print("det_2nd = ", dets_second)
        
        detections = self.init_track(dets, scores_keep, cls_keep, img)

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        """ Step 2: First association, with high score detection boxes"""
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        
        if hasattr(self, 'gmc'):
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            matcheddet = dets[idet]
            #print("matched det = ", matcheddet)
            #print("tracking id = ", int(track.track_id))
            matcheddetwithid = np.append(matcheddet, int(track.track_id))                                              # det with id
            tracking_id = int(track.track_id) - 1                                                                      # id number
            det1 =   self.trackslist[tracking_id]                                                                      # search previous 1st det according to id meet
            det2 = matcheddet                                                                                          # current or 2nd det
            p1 = [int((det1[0] + det1[2])/2) , int((det1[1] + det1[3])/2)]                                            
            p2 = [int((det2[0] + det2[2])/2) , int((det2[1] + det2[3])/2)]
            #print("p1 = ", p1)
            #print("p2 = ", p2)
            det2w = int(det2[2] - det2[0])
            det2h = int(det2[3] - det2[1])
            #print("Width and height = ", det2w, " , ", det2h)

            dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5                                                 # calculate distance
            #print("distance = ", dis)
            
            fx = abs(p1[0] - p2[0])                                                                                    # calculate x difference
            fy = abs(p1[1] - p2[1])                                                                                    # calculate y difference

            if (p1[0] > p2[0]):   
                fx = p2[0] - fx                                                                                        # calculate new x                                                                               
            else:
                fx = fx + p2[0]

            if (p1[1] > p2[1]):                                                                                        # calculate new y
                fy = p2[1] - fy
            else:
                fy = fy + p2[1]
            
            dis2 = ((int(fx) - p2[0]) ** 2 + (int(fy) - p2[1]) ** 2) ** 0.5                                            # calculate distance 2 for check
            #print("distance2 = ", dis2)

            #print("point3's x = ", int(fx))
            #print("point3's y = ", int(fy))
            det3 = list(det2).copy()                                                                                   # copy 2nd det for other info
            #print(type(det3))
            #print("original det = ", det3)
            det3[0] = int(int(fx)-(det2w/2))
            det3[1] = int(int(fy)-(det2h/2))
            det3[2] = int(int(fx)+(det2w/2))
            det3[3] = int(int(fy)+(det2h/2))                                                                           # create 3rd det
            det3idx = int(det3[4])
            
            if (int(det3[0]) < 0) or (int(det3[1]) < 0) or (int(det3[2]) > 1906) or (int(det3[3]) > 1080):             # deciding frame width/height limits
                #print("Some codes needed")
                if (int(det3[0]) < 0) and (int(det3[1]) < 0):
                    if (0-(int(det3[0])) > (det2w*2/3)) or (0-(int(det3[1])) > (det2h*2/3)):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif (int(det3[2]) > 1906) and (int(det3[3]) > 1080):
                    if ((int(det3[2]) - 1906) > (det2w*2/3)) or ((int(det3[3]) - 1080) > (det2h*2/3)):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif (int(det3[0]) < 0) and (int(det3[3]) > 1080):
                    if (0-(int(det3[0])) > (det2w*2/3)) or ((int(det3[3]) - 1080) > (det2h*2/3)):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif (int(det3[2]) > 1906) and (int(det3[1]) < 0):
                    if ((int(det3[2]) - 1906) > (det2w*2/3)) or (0-(int(det3[1])) > (det2h*2/3)):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif int(det3[0]) < 0:
                    if 0-(int(det3[0])) > (det2w*2/3):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif int(det3[1]) < 0:
                    if 0-(int(det3[1])) > (det2h*2/3):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif int(det3[2]) > 1906:
                    if (int(det3[2]) - 1906) > (det2w*2/3):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif int(det3[3]) > 1080:
                    if (int(det3[3]) - 1080) > (det2h*2/3):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
    
            else:
                print("Point 3  box = ", det3)
            #print("Point 3  box = ", det3)
            newdets = list(dets)                                                                                       # create list to use dets
            indeX = [i for i ,e in enumerate(newdets) if e[4] == det3idx]                                              # find index from id number
            newdets[int(indeX[0])] = det3                                                                              # insert/replace with 3rd det to this index place
            #print("Newdets = ", newdets)
            newdetections = self.init_track(newdets, scores_keep, cls_keep, img)                                       # apply newdets to create tracklets
            det = newdetections[idet]

            track3 = det1                                                                                              # apply to change tracklet info det1
            #print("original track = ", track3)
            track3[0] = det3[0]
            track3[1] = det3[1]
            track3[2] = det3[2]
            track3[3] = det3[3]                                                                                        # change with det3 values
            #print("New  3   track = ", track3)
            self.trackslist[tracking_id] = track3
            #print("trackslist = ", self.trackslist)
            
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            detwithid.append(list(matcheddetwithid))
        

        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        # TODO
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]

            matcheddet = dets_second[idet]
            #print("matched det = ", matcheddet)
            #print("tracking id = ", int(track.track_id))
            matcheddetwithid = np.append(matcheddet, int(track.track_id))
            tracking_id = int(track.track_id) - 1
            det1 =   self.trackslist[tracking_id]
            det2 = matcheddet
            p1 = [int((det1[0] + det1[2])/2) , int((det1[1] + det1[3])/2)]
            p2 = [int((det2[0] + det2[2])/2) , int((det2[1] + det2[3])/2)]
            #print("p1 = ", p1)
            #print("p2 = ", p2)
            det2w = int(det2[2] - det2[0])
            det2h = int(det2[3] - det2[1])
            #print("Width and height = ", det2w, " , ", det2h)

            dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            #print("distance = ", dis)
            
            fx = abs(p1[0] - p2[0])
            fy = abs(p1[1] - p2[1])

            if (p1[0] > p2[0]):
                fx = p2[0] - fx
            else:
                fx = fx + p2[0]

            if (p1[1] > p2[1]):
                fy = p2[1] - fy
            else:
                fy = fy + p2[1]
            
            dis2 = ((int(fx) - p2[0]) ** 2 + (int(fy) - p2[1]) ** 2) ** 0.5
            #print("distance2 = ", dis2)

            #print("point3's x = ", int(fx))
            #print("point3's y = ", int(fy))
            det3 = list(det2).copy()
            #print(type(det3))
            #print("original det = ", det3)
            det3[0] = int(int(fx)-(det2w/2))
            det3[1] = int(int(fy)-(det2h/2))
            det3[2] = int(int(fx)+(det2w/2))
            det3[3] = int(int(fy)+(det2h/2))
            det3idx = int(det3[4])
            
            if (int(det3[0]) < 0) or (int(det3[1]) < 0) or (int(det3[2]) > 1906) or (int(det3[3]) > 1080):
                #print("Some codes needed")
                if (int(det3[0]) < 0) and (int(det3[1]) < 0):
                    if (0-(int(det3[0])) > (det2w*2/3)) or (0-(int(det3[1])) > (det2h*2/3)):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif (int(det3[2]) > 1906) and (int(det3[3]) > 1080):
                    if ((int(det3[2]) - 1906) > (det2w*2/3)) or ((int(det3[3]) - 1080) > (det2h*2/3)):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif (int(det3[0]) < 0) and (int(det3[3]) > 1080):
                    if (0-(int(det3[0])) > (det2w*2/3)) or ((int(det3[3]) - 1080) > (det2h*2/3)):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif (int(det3[2]) > 1906) and (int(det3[1]) < 0):
                    if ((int(det3[2]) - 1906) > (det2w*2/3)) or (0-(int(det3[1])) > (det2h*2/3)):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif int(det3[0]) < 0:
                    if 0-(int(det3[0])) > (det2w*2/3):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif int(det3[1]) < 0:
                    if 0-(int(det3[1])) > (det2h*2/3):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif int(det3[2]) > 1906:
                    if (int(det3[2]) - 1906) > (det2w*2/3):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
                elif int(det3[3]) > 1080:
                    if (int(det3[3]) - 1080) > (det2h*2/3):
                        print("just use original")
                        det3.clear()
                        det3 = det2.copy()
                    else:
                        print("Point 3  box = ", det3)
    
            else:
                print("Point 3  box = ", det3)
            #print("Point 3  box = ", det3)
            newdetssecond = list(dets_second)
            indeX = [i for i ,e in enumerate(newdetssecond) if e[4] == det3idx]
            newdetssecond[int(indeX[0])] = det3
            #print("Newdets = ", newdetssecond)
            newdetections_second = self.init_track(newdetssecond, scores_second, cls_second, img)
            det = newdetections_second[idet]
            
            track3 = det1
            #print("original track = ", track3)
            track3[0] = det3[0]
            track3[1] = det3[1]
            track3[2] = det3[2]
            track3[3] = det3[3]
            #print("New  3   track = ", track3)
            self.trackslist[tracking_id] = track3
            #print("trackslist = ", self.trackslist)

            
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            detwithid.append(list(matcheddetwithid))


        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dets = [dets[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            uncondet = dets[idet]
            uncondet = np.append(uncondet, int((unconfirmed[itracked]).track_id))
            detwithid.append(list(uncondet))
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:                                      #################
                continue
            self.trackingid += 1
            newdet = dets[inew]
            newdet = np.append(newdet, int(self.trackingid))
            print("New det = ", newdet)
            self.trackslist.append(list(newdet))
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            detwithid.append(list(newdet))

        print("detwithid = ", detwithid)
        #print(self.trackslist)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        '''
        if self.frame_id == 1:
            output = [track.tlbr.tolist() + [track.track_id, track.score, track.cls, track.idx] for track in self.tracked_stracks if track.is_activated]
        else:
            output = []
            for track in self.tracked_stracks:
                a = []
                if track.is_activated:
                    for i in detwithid:
            	        a = i[0:4] if i[5] == int(track.track_id) else a
                    output.append(a + [track.track_id, track.score, track.cls, track.idx])'''
        output = []
        for track in self.tracked_stracks:
            a = []
            if track.is_activated:
                for i in detwithid:
            	    a = i[0:4] if i[5] == int(track.track_id) else a
                output.append(a + [track.track_id, track.score, track.cls, track.idx])
        
        print("output = ", output)
        return np.asarray(output, dtype=np.float32)

    def get_kalmanfilter(self):
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # detections

    def get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)
        # TODO: mot20
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        STrack.multi_predict(tracks)

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
