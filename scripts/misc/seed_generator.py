import numpy as np


def generate_seed(bits_count: int):
    ones_count = bits_count // 2
    zeros_count = bits_count - ones_count

    bits_list = ["1"] * ones_count + ["0"] * zeros_count
    np.random.shuffle(bits_list)

    bits_string = "".join(bits_list)
    seed = int(bits_string, 2)

    return seed


def main():
    seeds = [generate_seed(bits_count=32) for _ in range(10)]
    print("Generated seeds:")
    for seed in seeds:
        print(seed)


if __name__ == "__main__":
    main()
