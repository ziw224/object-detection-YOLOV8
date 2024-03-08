import math

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
        # Optional: Store additional object info (e.g., class IDs)
        self.object_info = {}

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h, d= rect
            conf = rect[5] if len(rect) > 5 else None  # Handle optional confidence score

            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id, d])  # Include class ID
                    self.object_info[id] = {'class_id': d, 'confidence': conf}  # Store additional info
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
                objects_bbs_ids.append([x, y, w, h, self.id_count, d])  # Include class ID
                self.object_info[self.id_count] = {'class_id': d, 'confidence': conf}  # Store additional info
                self.id_count += 1

        return objects_bbs_ids