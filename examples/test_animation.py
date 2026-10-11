import numpy as np

from mechacode.mechanism import LegRestMechanism
single_choice = np.array([
            [
                36.539245982427595,
                28.024617410140323
            ],
            [
                35.87942545834548,
                61.612532611381255
            ],
            [
                36.30424776629044,
                111.28006691119293
            ],
            [
                38.32926341115385,
                93.40580470804511
            ],
            [
                43.39362579184341,
                26.051947111474092
            ],
            [
                42.43135186300394,
                113.54397036595957
            ],
            [
                43.96935685036926,
                62.258478880845594
            ]
        ])

def main():
    my_legrest = LegRestMechanism(single_choice)
    my_legrest.plot_configuration()
    my_legrest.animate_mechanism([np.pi/2 - my_legrest.link_angles[3], np.pi/2])


if __name__ == "__main__":
    main()