

from os import system


def main():
    print("Extending the dataset...")
    system("python extend_dataset.py")
    print("Learning...")
    system("python tf_aerial_images.py -r")
    print("Creating submission")
    system("python mask_to_submission.py")

if __name__ == '__main__':
    main()
