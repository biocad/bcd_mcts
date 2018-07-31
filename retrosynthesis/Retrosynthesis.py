import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mcts.MCTS import *


class RetroNode(TreeNode):
    def __init__(self, parent_edge, **kwargs):

        """
         Implementation of Monte-Carlo Tree Node related to retrosynthetic tree.
         :param parent_edge: TreeEdge or None if the node is root
         :param model: keras model
         :param mols: molecule for retrosynthesis
         :param home_mols: decomposed molecules that the company has
         :param available_substances: list of molecules that the company has
         :param re_types: reaction types
         :param depth: node depth
         :param mols: molecules that are not decomposed to molecules that the company has

         """

        super().__init__(parent_edge)

        self.children = []
        self.model = kwargs["model"]
        self.mols = kwargs["mols"]
        self.home_mols = kwargs["home_mols"]
        self.available_substances = kwargs["available_substances"]
        self.re_types = kwargs["re_types"]
        self.depth = kwargs["depth"]
        self.home_mols = self.home_mols + [m for m in self.mols if m in self.available_substances]
        self.mols = list(set(self.mols) - set(self.home_mols))
        self.fp_size = 128
        self.max_depth = 5

    def _get_fp(self):
        mols_fp = []
        for mol in self.mols:
            m = Chem.MolFromSmiles(mol)
            AllChem.AddHs(m)
            Chem.SanitizeMol(m)
            cur_fp_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=self.fp_size)
            cur_fp = cur_fp_.ToBitString()
            mols_fp.append(cur_fp)
        return mols_fp

    def get_available_child_nodes(self):
        if self.get_reward() != 0:
            return []

        results = []
        data_for_modeling = self._get_fp()
        list_of_fp = []
        for fingerprint in data_for_modeling:
            count = -1
            fingerprint_dict = dict()
            for elem_of_fp in fingerprint:
                count += 1
                fingerprint_dict[count] = elem_of_fp
            list_of_fp.append(fingerprint_dict)
        df_list_of_fp = pd.DataFrame(list_of_fp).astype(np.int8)
        pred = self.model.predict(df_list_of_fp.values)
        all_predictions = []
        for IDX in range(len(pred)):
            non_zero_pred = np.where(pred[IDX, :] > 0.01)[0]
            sorted_non_zero_idx = non_zero_pred[np.argsort(pred[IDX, non_zero_pred])][-5:]
            sorted_probas = pred[IDX, sorted_non_zero_idx][-5:]
            all_predictions.append(list(zip(sorted_non_zero_idx, sorted_probas)))

        all_new = []
        for i in range(len(all_predictions)):
            list_of_smiles = [ll for ii, ll in enumerate(self.mols) if ii != i]
            s = list(zip(all_predictions[i], [list_of_smiles] * len(all_predictions[i])))
            all_new.append(s)

        for cur_mol, cur_decomposition in list(zip(self.mols, all_new)):
            cur_mol = Chem.MolFromSmiles(cur_mol)

            for elem in cur_decomposition:
                r_type = elem[0][0]
                cur_smirks = self.re_types.loc[r_type].mapped_SMILES
                pred_rea = AllChem.ReactionFromSmarts(">>")
                pred_rea.AddReactantTemplate(Chem.MolFromSmarts(cur_smirks.split(">>")[1]))
                pred_rea.AddProductTemplate(Chem.MolFromSmarts(cur_smirks.split(">>")[0]))
                pred_rea.Initialize()

                pred_rcts_mol = None
                for d in pred_rea.RunReactants([AllChem.AddHs(cur_mol)]):
                    try:
                        pred_rcts_mol = d[0]
                        Chem.SanitizeMol(d[0], Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
                    except ValueError:
                        continue
                if pred_rcts_mol is None:
                    continue

                Chem.SanitizeMol(pred_rcts_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
                mol_children = Chem.MolToSmiles(AllChem.RemoveHs(pred_rcts_mol))
                everything_for_children = mol_children.split(".")
                new_node = RetroNode(None, home_mols=self.home_mols, depth=self.depth + 1, model=self.model,
                                     mols=everything_for_children, available_substances=self.available_substances, re_types=self.re_types)
                results.append((new_node, 1))

        return results

    def get_reward(self):
        if self.depth == self.max_depth:
            return -1
        elif len(self.mols) == 0:
            return 1
        else:
            return 0

    def expand(self):
        super().expand()

    def __eq__(self, other):
        return self == other

    def pretty_print(self):
        return ""


