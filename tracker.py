import math
import cv2

class Trajectory:
    def __init__(self, max_length=10):
        self.positions = []
        self.max_length = max_length

    def add_position(self, pos):
        self.positions.append(pos)
        if len(self.positions) > self.max_length:
            self.positions.pop(0)

    def get_positions(self):
        return self.positions

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.trajectories = {}  # Store trajectories for each object

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    # Add position to trajectory
                    if id not in self.trajectories:
                        self.trajectories[id] = Trajectory()
                    self.trajectories[id].add_position((cx, cy))
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.trajectories[self.id_count] = Trajectory()  # Create a new trajectory for the new object
                self.trajectories[self.id_count].add_position((cx, cy))
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        new_trajectories = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
            # Maintain trajectories for active objects
            new_trajectories[object_id] = self.trajectories[object_id]

        self.center_points = new_center_points.copy()
        self.trajectories = new_trajectories
        return objects_bbs_ids

    def draw_trajectories(self, frame):
        for trajectory in self.trajectories.values():
            positions = trajectory.get_positions()
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i - 1], positions[i], (0, 0, 255), 2)