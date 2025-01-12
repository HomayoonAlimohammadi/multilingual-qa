from train import train
from test import test

def main():
    trainer = train()
    test(trainer)

if __name__ == "__main__":
    main()
