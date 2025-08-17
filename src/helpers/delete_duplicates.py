import os
import re

def delete_numbered_pdfs(folder_path):
    # Regex pattern for files like (number).pdf
    pattern = re.compile(r"\(\d+\)\.pdf$", re.IGNORECASE)

    # Get matching files
    matching_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and pattern.search(f)
    ]

    # Print the count
    print(f"Found {len(matching_files)} matching files.")

    if not matching_files:
        return  # Nothing to delete

    # Delete files
    deleted_files = []
    for filename in matching_files:
        file_path = os.path.join(folder_path, filename)
        try:
            os.remove(file_path)
            deleted_files.append(filename)
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

    # Write 'deletes' file in the same folder
    deletes_file_path = os.path.join(folder_path, "deletes.txt")
    try:
        with open(deletes_file_path, "w", encoding="utf-8") as f:
            f.write("Deleted files:\n")
            for file in deleted_files:
                f.write(file + "\n")
        print(f"Deleted files list saved to: {deletes_file_path}")
    except Exception as e:
        print(f"Error writing deletes file: {e}")


if __name__ == "__main__":
    delete_numbered_pdfs("images/drawings_pdf")
