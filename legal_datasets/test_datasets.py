from data_utils import load_justice_data

if __name__ == "__main__":
    cases = load_justice_data()
    print(f"Total cases: {len(cases)}")
    print("First case details:")
    print(cases[0])
