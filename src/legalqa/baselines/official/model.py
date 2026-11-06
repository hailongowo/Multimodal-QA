"""Load the Visualized-BGE model used only by the official baseline."""


def load_model(model_weight):
    from visual_bge.modeling import Visualized_BGE

    model = Visualized_BGE(model_name_bge="BAAI/bge-m3", model_weight=str(model_weight))
    model.eval()
    return model
