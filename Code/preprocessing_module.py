import os
from torch.utils.data import Dataset
from skimage import io
from sklearn.model_selection import train_test_split

class BeanTechDataset(Dataset):
    """BeanTech dataset."""

    def __init__(self, images_arrays_list, images_labels_list, images_names_list, transform=None):
        """
        Args:
            images_arrays_list (list): List containing all images represented as numpy arrays.
            images_labels_list (list): List containing all images labels.
            images_names_list (list): List containing all images names.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_arrays_list = images_arrays_list
        self.images_labels_list = images_labels_list
        self.images_names_list = images_names_list
        self.transform = transform

    def __len__(self):
        return len(self.images_names_list)

    def __getitem__(self, idx):
        image = self.images_arrays_list[idx]
        image_label = self.images_labels_list[idx]
        image_name = self.images_names_list[idx]
        sample = {'image': image, 'image_label': image_label, 'image_name': image_name}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


def process_original_data(path_to_folder, dataset_name, images_type: float):
    # create empty lists
    names_all, names_albedo, names_other, names_east, names_north, names_south, names_west = ([] for i in range(7))
    arrays_all, arrays_albedo, arrays_other, arrays_east, arrays_north, arrays_south, arrays_west = ([] for i in range(7))
    # list of files in the folder
    files_list = os.listdir(path_to_folder)
    # sort the list alphabetical order
    files_list.sort()
    # reverse the list (pop() method removes and returns the last item)
    files_list.reverse()

    def load_image(direction_names_list, direction_images_list):
        if dataset_name == 'all':
            names_all.append(file_name)
            arrays_all.append(io.imread(path_to_folder + '/' + file_name)/255.0)
        elif dataset_name == 'albedo_nsew' or dataset_name == 'nsew':
            names_other.append(file_name)
            arrays_other.append(io.imread(path_to_folder + '/' + file_name)/255.0)
        elif dataset_name == 'albedo_nsew_split':
            direction_names_list.append(file_name)
            direction_images_list.append(io.imread(path_to_folder + '/' + file_name)/255.0)
        elif dataset_name == 'albedo':
            return

    # process the files
    while files_list:
        # albedo
        file_name = files_list.pop()
        numeric_file_id = file_name.split('_')[0]
        if dataset_name == 'all':
            names_all.append(file_name)
            arrays_all.append(io.imread(path_to_folder + '/' + file_name)/255.0)
        elif dataset_name == 'nsew':
            break
        else:
            names_albedo.append(file_name)
            arrays_albedo.append(io.imread(path_to_folder + '/' + file_name)/255.0)

        # east (same numeric id)
        file_name = numeric_file_id + '_East.bmp'
        files_list.pop(files_list.index(file_name))
        load_image(names_east, arrays_east)
        # north (same numeric id)
        file_name = numeric_file_id + '_North.bmp'
        files_list.pop(files_list.index(file_name))
        load_image(names_north, arrays_north)
        # south (same numeric id)
        file_name = numeric_file_id + '_South.bmp'
        files_list.pop(files_list.index(file_name))
        load_image(names_south, arrays_south)
        # west (same numeric id)
        file_name = numeric_file_id + '_West.bmp'
        files_list.pop(files_list.index(file_name))
        load_image(names_west, arrays_west)

    # create datasets
    if dataset_name == 'all':
        labels = [float(images_type) for i in range(len(names_all))]
        dataset_all = BeanTechDataset(arrays_all, labels, names_all)
        return [arrays_all], [labels], [names_all]
    elif dataset_name == 'albedo_nsew':
        labels_albedo = [float(images_type) for i in range(len(names_albedo))]
        labels_other = [float(images_type) for i in range(len(names_other))]
        array = [arrays_albedo, arrays_other]
        array_names = [names_albedo, names_other]
        array_labels = [labels_albedo, labels_other]
        return array, array_labels, array_names
    elif dataset_name == 'albedo_nsew_split':
        labels = [float(images_type) for i in range(len(names_albedo))]
        array = [arrays_albedo, arrays_east, arrays_north, arrays_south, arrays_west]
        array_names = [names_albedo, names_east, names_north, names_south, names_west]
        array_labels = [labels for i in range(5)]
        return array, array_labels, array_names
    elif dataset_name == 'albedo':
        labels = [float(images_type) for i in range(len(names_albedo))]
        return [arrays_albedo], [labels], [names_albedo]
    elif dataset_name == 'nsew':
        labels = [float(images_type) for i in range(len(names_other))]
        return [arrays_other], [labels], [names_other]


def build_datasets(path_to_folder_ok, path_to_folder_ko, dataset_name, transform=None, split=False):
    """
    Processes the original data and returns the same data organized in one or more BeanTechDataset object(s):\n
    - one BeanTechDataset object containing all the data if dataset_name = 'all'
    - one BeanTechDataset object containing only albedo data if dataset_name = 'albedo'
    - one BeanTechDataset object containing only raw data if dataset_name = 'nsew'
    - two BeanTechDataset objects (one for 'Albedo' images, one for all other images) if dataset_name = 'albedo_nsew';
    - five BeanTechDataset objects (one for 'Albedo' images, one for 'East' images, one for 'North' images, one for
      'South' images, one for 'West' images) if dataset_name = 'albedo_nsew_split'. iF split is true, it returns a list
      containing two tuples with the number of datasets specified above.

    :param path_to_folder_ok: path to the folder containing the ok images
    :type path_to_folder_ok: string
    :param path_to_folder_ko: path to the folder containing the ko images
    :type path_to_folder_ko: string
    :param dataset_name: name of the dataset
    :type dataset_name: string
    :param tranform: tranformation to apply
    :type images_type: float
    :param split: if true splits ok and ko data
    :type images_type: bool
    """
    array_ok, array_labels_ok, array_names_ok = process_original_data(path_to_folder_ok, dataset_name, 0.0)
    array_ko, array_labels_ko, array_names_ko = process_original_data(path_to_folder_ko, dataset_name, 1.0)
    if split:
        datasets_ok = build_datasets_from_array(array_ok, array_labels_ok, array_names_ok, transform)
        datasets_ko = build_datasets_from_array(array_ko, array_labels_ko, array_names_ko, transform)
        return [datasets_ok, datasets_ko]
    else:
        array = merge_lists_array(array_ok, array_ko)
        array_labels = merge_lists_array(array_labels_ok, array_labels_ko)
        array_names = merge_lists_array(array_names_ok, array_names_ko)
        return build_datasets_from_array(array, array_labels, array_names, transform)


def build_datasets_from_array(array, array_labels, array_names, transform):
    datasets = ()
    for images, labels, names in zip(array, array_labels, array_names):
        datasets += (BeanTechDataset(images, labels, names, transform),)
    return datasets


def split(dataset: BeanTechDataset, test_size, random_state = None):
    x_train, x_test, y_train, y_test, names_train, names_test = train_test_split(dataset.images_arrays_list,
                                                                                 dataset.images_labels_list,
                                                                                 dataset.images_names_list,
                                                                                 test_size=test_size,
                                                                                 random_state= random_state)

    return BeanTechDataset(x_train, y_train, names_train, dataset.transform), BeanTechDataset(x_test, y_test, names_test, dataset.transform)


def merge_lists_array(array1, array2):
    concat = []
    for list1, list2 in zip(array1, array2):
        concat.append(list1+list2)
    return concat

