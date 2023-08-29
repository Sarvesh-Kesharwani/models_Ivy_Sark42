input_file_path = "/workspaces/models_Ivy_Sark42/ivy_models/googlenet/layers.py"
output_file_path = "m_" + input_file_path


def remove_lines_starting_with(prefix, input_file, output_file):
    with open(input_file, "r") as input_f, open(output_file, "w") as output_f:
        for line in input_f:
            if not line.strip().startswith(prefix):
                output_f.write(line)


prefix_to_remove = "pf("
remove_lines_starting_with(prefix_to_remove, input_file_path, output_file_path)
print("Lines starting with '{}' removed.".format(prefix_to_remove))
