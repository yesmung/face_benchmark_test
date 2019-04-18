import os, sys
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description=
        "This example script will show you how to use the face detector module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default='mobilenetv2_ssdlite',
        help=
        "Please choose between : opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd, mobilenetv2_ssdlite "
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Please output root path")

    args = parser.parse_args()
    return args

def create_output_folder(dirname, output_path) :
    try :
        filenames = os.listdir(dirname)
        for filename in filenames : 
            try :
                out = os.path.join(output_path, filename)
                print(out)
                if not (os.path.isdir(out)) :
                    os.makedirs(out)
            except OSError as e :
                if e.errno != errno.EEXIST:
                    print("Failed to create directory!!!")
                    raise
    except PermissionError :
        pass

filename_list = []
classname_list = []
file_dic = {}

def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
        	full_filename = os.path.join(dirname, filename)
        	if os.path.isdir(full_filename):
        		search(full_filename)
        	else:
        		ext = os.path.splitext(full_filename)[-1]
        		print(full_filename)
    except PermissionError:
        pass


def main() :
 	args = get_args()

 	# create_output_folder(args.input, args.output)
 	search(args.input)

if __name__ == '__main__':
    main()