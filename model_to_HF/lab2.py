import os

# Specify the path to the folder you want to list files from
folder_path = "/workspaces/models_Ivy_Sark42/ivy_models"

# Use os.listdir() to get a list of all items in the folder
items = os.listdir(folder_path)

# Use a list comprehension to filter out only the subfolders
subfolders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

# Print the list of subfolders
model_list = []
for subfolder in subfolders:
    model_list.append(subfolder)

print(model_list)
# # Use os.listdir() to get a list of all files and directories in the folder
# files = os.listdir(folder_path)

# # Loop through the list of files and print their names
# for file in files:
#     # Check if the path is a file (not a directory)
#     if os.path.isfile(os.path.join(folder_path, file)):
#         print(file)
