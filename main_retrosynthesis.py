import argparse
from retrosynthesis.Retrosynthesis import *
from keras.models import load_model
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mols", type=str, help="canonical SMILES of molecule for prediction", required=True)
    parser.add_argument("-p_m", "--path_to_mod", type=str, help="path to model for predict", default="data/1_11.ckpt")
    parser.add_argument("-r_t", "--re_types", type=str, help="path to file with types of reactions",
                        default="data/reaction_types.csv")
    parser.add_argument("-db", "--av_sub", type=str, help="path to file with available substances",
                        default="data/ChemBase")
    parser.add_argument("-s", "--pdf_name", type=str, help="path to output pdf file", required=True)
    args = vars(parser.parse_args())
    smi_type_df = pd.read_csv(args["re_types"])
    smi_type_df.columns = ["unmapped_SMILES", "mapped_SMILES"]
    mols = args["mols"]
    re_types = args["re_types"]
    av_sub = args["av_sub"]
    model = load_model(args["path_to_mod"])
    target_mol = [Chem.CanonSmiles(mols)]
    file_with_substance = open(av_sub).read().split("\n")

    root_retro_node = RetroNode(None, mols=target_mol, model=model,
                                available_substances=file_with_substance,
                                re_types=smi_type_df, depth=1, home_mols=[])
    cur_node = root_retro_node
    tree_of_retro = MonteCarloTree(root_retro_node)
    tree_of_retro.fit()
    list_of_nodes = MonteCarloTree.traverse(tree_of_retro)

    list_of_sorted_nodes = []
    for n in list_of_nodes:
        if n.parent_edge != None:
            if (len(n.children) == 0) and (n.parent_edge.value > 0):
                list_of_sorted_nodes.append(n)
    if len(list_of_nodes) == 1:
        print("Molecules that the company has")
        sys.exit()
    elif len(list_of_sorted_nodes) == 0:
        print("I can't help you :(")
        sys.exit()
    pdf_pages = PdfPages(args["pdf_name"])
    react_all_branch = []
    all_ways_for_mol = []
    plt.ioff()
    for j in range(len(list_of_sorted_nodes)):
        fig = plt.figure(figsize=(36, 60))
        cur_node = list_of_sorted_nodes[j]
        all_mols = []
        while cur_node.parent_edge is not None:
            all_mols.append((cur_node.home_mols + cur_node.mols))
            cur_node = cur_node.parent_edge.parent
        all_mols.append((cur_node.home_mols + cur_node.mols))
        reagents_list = []
        products_list = []
        for cur_mols_list, par_mols_list in zip(all_mols, all_mols[1:]):
            cur_set = set(cur_mols_list)
            par_set = set(par_mols_list)
            reagents_list.append(list(cur_set - par_set))
            products_list.append(list(par_set - cur_set))

        prep_reagents_list = [".".join([Chem.MolToSmiles(AllChem.MolFromSmiles(mol), allBondsExplicit=True, isomericSmiles=True) for mol in cur_list])
                              for cur_list in reagents_list]
        prep_products_list = [".".join([Chem.MolToSmiles(AllChem.MolFromSmiles(mol), allBondsExplicit=True, isomericSmiles=True) for mol in cur_list])
                              for cur_list in products_list]

        reactions_list = []
        for i in range(len(prep_reagents_list)):
            reaction = prep_reagents_list[i] + ">>" + prep_products_list[i]
            reactions_list.append(reaction)

        for string, reaction in enumerate(reactions_list):
            ax = plt.subplot2grid((9, 1), (2 + string, 0), colspan=1, rowspan=1)
            a = AllChem.ReactionFromSmarts(reaction)
            ax.imshow(Draw.ReactionToImage(a, subImgSize=(300, 150)), interpolation='bilinear')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.axis('off')
        ax = plt.subplot2grid((9, 1), (0, 0), colspan=1, rowspan=2)
        plt.axis('off')
        ax.imshow(Draw.MolToImage(Chem.MolFromSmiles(target_mol[0]), size=(600, 300)), interpolation="bilinear")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pdf_pages.savefig(fig)
    pdf_pages.close()
