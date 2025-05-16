# command ในนี้ อิงตาม flowchart นะครับ
#import library
import os

def rename_files_in_folder(folder_path, prefix):
    # read all file from path
    files = os.listdir(folder_path)
    # sort filename alphabetically
    files.sort()  
    # Loop through each file with index i (start from 1)
    for i, filename in enumerate(files, start=1):
        # split filename into name and extention
        ext = os.path.splitext(filename)[1]  
        # Create new filename using prefix + "" + i + ext
        new_name = f"{prefix}_{i}{ext}"
        
        # join path to get full old_file and new file 
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # rename old_file -> new_file
        os.rename(old_file, new_file)
        # Print rename message 
        print(f"Renamed: {filename} -> {new_name}")

# call funtion to rename data 
rename_files_in_folder("data/happy", "happy")
rename_files_in_folder("data/sad", "sad")
