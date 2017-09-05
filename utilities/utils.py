import json


def load_data(im_type, dataset='bottles', q=1):
    """
    functon to load database/query names, descriptors, and keypoints
    """
    image_names_file = None
    descriptors_file = None
    keypoints_file = None
    if im_type == 'database':
        image_names_file = 'image_names.json'
        descriptors_file = 'descriptors.json'
        keypoints_file = 'keypoints.json'
    elif im_type == 'query':
        image_names_file = 'q{}_image_names.json'.format(q)
        descriptors_file = 'q{}_descriptors.json'.format(q)
        keypoints_file = 'q{}_keypoints.json'.format(q)
    else:
        print "im_type must be one of 'database' or 'query'"
    with open('data/' + dataset + '/' + image_names_file, 'r') as f:
        image_names = json.load(f)
    with open('data/' + dataset + '/' + descriptors_file, 'r') as f:
        image_descriptors = json.load(f)
    with open('data/' + dataset + '/' + keypoints_file, 'r') as f:
        image_keypoints = json.load(f)
    return (image_names, image_descriptors, image_keypoints)
