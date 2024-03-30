
#!/bin/bash

# Activate the conda environment


# Loop through each line in requirements.txt
while IFS= read -r line; do
    # Try to install the package, if it fails, catch the error and print the package name
    pip install "$line" || echo "Failed to install $line, skipping..."
done < requirements.txt
