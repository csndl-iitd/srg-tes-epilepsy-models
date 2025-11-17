from tes.data_ingestion import DataRead

def main():
    reader = DataRead()  # create object of the class
    reader.new_algo_ext(
        25,
        "raw/workstation/tes_enti_go3.bz2",
        "raw/workstation/tes_enti_lo3.bz2"
    )

if __name__ == "__main__":
    main()
