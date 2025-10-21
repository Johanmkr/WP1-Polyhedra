from arg_parser import parse_and_merge_config
def main():
    # Get the final configuration
    config = parse_and_merge_config()
    # Use the configuration (e.g., start the experiment)
    print("Final Configuration:")
    print(config)
if __name__ == "__main__":
    main()
