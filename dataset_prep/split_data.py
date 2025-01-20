import random
import sys
sys.path.append('/home/mij17663/code')
import utils.file_handling as fh

def pick_random_numbers(amount, start, end):
    """
    Randomly pick a given amount of unique numbers from a specified range.

    Parameters:
    - amount (int): The number of unique numbers to pick.
    - start (int): The start of the range (inclusive).
    - end (int): The end of the range (inclusive).

    Returns:
    - list: A list of randomly picked unique numbers.
    """
    if amount > (end - start + 1):
        raise ValueError("Amount exceeds the number of available unique numbers in the range.")
    
    return random.sample(range(start, end + 1), amount)

# Example usage
# amount = 27
# start = 1
# end = 267
# random_numbers = pick_random_numbers(amount, start, end)
# print(f"Randomly picked numbers: {random_numbers}")
# Randomly picked numbers: [219, 89, 129, 50, 58, 192, 144, 178, 132, 174, 116, 68, 105, 224, 71, 25, 222, 72, 202, 20, 251, 8, 195, 40, 54, 233, 84]

numbers = [219, 89, 129, 50, 58, 192, 144, 178, 132, 174, 116, 68, 105, 224, 71, 25, 222, 72, 202, 20, 251, 8, 195, 40, 54, 233, 84]
# Format each number to have four digits with leading zeros
formatted_numbers = [f"{num:04d}" for num in numbers]
# ['0219', '0089', '0129', '0050', '0058', '0192', '0144', '0178', '0132', '0174', '0116', '0068', '0105', '0224', '0071', '0025', '0222', '0072', '0202', '0020', '0251', '0008', '0195', '0040', '0054', '0233', '0084']

print(formatted_numbers)

src_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/AC_PET_Tr'
dst_dir = '/mnt/data/mij17663/nac2ac_sep/2d_slices/AC_PET_Ts'

for i in formatted_numbers:
    pattern = f'Nac2Ac{i}'
    print(f"Moving files with pattern: {pattern}")
    fh.move_spec_files(src_dir, dst_dir, pattern)