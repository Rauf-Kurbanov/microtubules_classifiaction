import pandas as pd
from os import listdir
import os


def main():
    HUMAN_JPG_PATH = "/Users/raufkurbanov/Programs/microtubules_classifiaction/data/to_deploy"
    base = "https://host-data--rauf-kurbanov.jobs.neuro-public.org.neu.ro/data"

    for test_dir, save_path in [("FilteredCleanProcessed/test", "test_links.csv")]:
        test_base = os.path.join(base, test_dir)

        links = [os.path.join(test_base, x) for x in
                     listdir(os.path.join(HUMAN_JPG_PATH, test_dir))]
        links_df = pd.DataFrame({"image link": links})
        links_df.to_csv(save_path, index=None)


if __name__ == '__main__':
    main()
