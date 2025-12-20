from tes.data_ingestion import DataRead

def main():
    reader = DataRead()  # create object of the class
    df_loc=reader.data_read_bz2('raw/workstation/rep_mos_lo.bz2')
    # reader.new_algo_ext(
    #     200,
    #     "raw/workstation/rep_mos_go.bz2",
    #     "raw/workstation/rep_mos_lo.bz2"
    # )
    reader.ext_node_community(df_loc,0.5,"raw/workstation/rep_mos_nc.bz2")

if __name__ == "__main__":
    main()
