import json
import matplotlib.pyplot as plt

def main():
    with open('results.json') as f:
        results = json.load(f)

    # keys
    print(results.keys())

    # loss
    plt.plot(results['loss'])
    plt.ylabel('loss')
    plt.show()

    # reward
    plt.plot(results['episode_reward'])
    plt.ylabel('episode_reward')
    plt.show()

    # steps
    plt.plot(results['nb_steps'])
    plt.ylabel('nb_steps')
    plt.show()

if __name__ == "__main__":
    main()
