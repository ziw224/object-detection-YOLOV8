import os
import pickle
import numpy as np
import random

class DataLoader():

    def __init__(self, batch_size=50, seq_length=5, datasets=[0, 1, 2, 3, 4], forcePreProcess=False):
        # List of data directories where raw data resides
        self.data_dirs = ['./data/new']
        # self.data_dirs = ['./data/eth/univ', './data/eth/hotel']

        try:
            self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        except IndexError as e:
            print(f"An index in 'datasets' is out of range: {e}")
            # Handle the error appropriately, maybe by setting a default directory or halting execution gracefully.

        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data'

        # Store the batch size and the sequence length arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path of the file in which the data needs to be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist already or if forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files
            self.preprocess(self.used_data_dirs, data_file)

        # Load the data from the pickled file
        self.load_preprocessed(data_file)
        # Reset all the pointers
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        all_vehicle_data = {}
        dataset_indices = []
        current_vehicle = 0
        # For each dataset
        for directory in data_dirs:
            # Adjust the file path for your new CSV files
            file_path = os.path.join(directory, 'vehicle_data_1.csv')
            print("Processing {}".format(file_path))

            # Load data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')

            # Assuming the first column is the object ID
            unique_vehicles = np.unique(data[:, 0])

            for vehicle_id in unique_vehicles:
                # Filter data for the current vehicle
                vehicle_data = data[data[:, 0] == vehicle_id, :]

                # Calculate the center of the bounding box
                x_center = (vehicle_data[:, 2] + vehicle_data[:, 3]) / 2
                y_center = (vehicle_data[:, 4] + vehicle_data[:, 5]) / 2

                # Assuming the second column is the frame number
                frame_ids = vehicle_data[:, 1]

                # Combine the new x, y centers with their frame IDs
                traj = np.vstack((frame_ids, x_center, y_center)).T

                # Store this in the dictionary
                all_vehicle_data[current_vehicle + int(vehicle_id)] = traj

            # Current dataset done
            dataset_indices.append(current_vehicle + len(unique_vehicles))
            current_vehicle += len(unique_vehicles)

            print("Total vehicle numbers: {}".format(len(unique_vehicles)))

        # The complete data is a tuple of all vehicle data and dataset vehicle indices
        complete_data = (all_vehicle_data, dataset_indices)
        # Store the complete data into the pickle file
        f = open(data_file, "wb")
        pickle.dump(complete_data, f, protocol=2)
        f.close()
    def load_preprocessed(self, data_file):
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()

        # Get the pedestrian data from the pickle file
        all_ped_data = self.raw_data[0]
        # Not using dataset_indices for now
        # dataset_indices = self.raw_data[1]

        # Construct the data with sequences(or trajectories) longer than seq_length
        self.data = []
        counter = 0

        # For each pedestrian in the data
        for ped in all_ped_data:
            # Extract his trajectory
            traj = all_ped_data[ped]
            # If the length of the trajectory is greater than seq_length (+2 as we need both source and target data)
            if traj.shape[1] > (self.seq_length+2):
                # TODO: (Improve) Store only the (x,y) coordinates for now
                self.data.append(traj[[0, 1], :].T)
                # Number of batches this datapoint is worth
                counter += int(traj.shape[1] / ((self.seq_length+2)))

        print("all ped data len: {}, seq length: {}".format(len(all_ped_data), self.seq_length))
        # Calculate the number of batches (each of batch_size) in the data
        self.num_batches = int(counter / self.batch_size)

    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        # List of source and target data for the current batch
        x_batch = []
        y_batch = []
        # For each sequence in the batch
        for i in range(self.batch_size):
            # Extract the trajectory of the pedestrian pointed out by self.pointer
            traj = self.data[self.pointer]
            # Number of sequences corresponding to his trajectory
            n_batch = int(traj.shape[0] / (self.seq_length+2))
            # Randomly sample a index from which his trajectory is to be considered
            idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
            # Append the trajectory from idx until seq_length into source and target data
            x_batch.append(np.copy(traj[idx:idx+self.seq_length, :]))
            y_batch.append(np.copy(traj[idx+1:idx+self.seq_length+1, :]))

            if random.random() < (1.0/float(n_batch)):
                # Adjust sampling probability
                # if this is a long datapoint, sample this data more with
                # higher probability
                self.tick_batch_pointer()

        return x_batch, y_batch

    def tick_batch_pointer(self):
        '''
        Advance the data pointer
        '''
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset the data pointer
        '''
        self.pointer = 0