from utils import *
from transformers import AutoModel, AutoTokenizer

MAXLEN = 128


def mk_tensors(txt, tokenizer, maxlen):
    tok_res = tokenizer(txt, truncation=True, padding=True, max_length=maxlen)
    input_ids = tok_res["input_ids"]
    token_type_ids = tok_res["token_type_ids"]
    attention_mask = tok_res["attention_mask"]
    input_ids = torch.tensor(input_ids)
    token_type_ids = torch.tensor(token_type_ids)
    attention_mask = torch.tensor(attention_mask)
    return input_ids, attention_mask, token_type_ids


def mk_dl(tensors):
    input_ids, token_type_ids, attention_mask = tensors
    ds = TensorDataset(input_ids, attention_mask, token_type_ids)
    dl = DataLoader(
        dataset=ds,
        batch_size=BS,
        num_workers=NWKRS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    return dl


def mk_feats(df, nm_mdl):
    device = torch.device(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(nm_mdl, do_lower_case=False)
    tensors = mk_tensors(list(df.title.values), tokenizer, maxlen=MAXLEN)
    dl = mk_dl(tensors)
    mdl = AutoModel.from_pretrained(nm_mdl).to(device)
    mdl.eval()
    feats = np.zeros((len(df), 768))
    i = 0
    for dat in dl:
        with torch.no_grad():
            dat = (o.to(device) for o in dat)
            output = mdl(*dat)
            fts = output.last_hidden_state
            fts = fts.mean(dim=-2)
            l = len(fts)
            feats[i : i + l, :] = fts.cpu().detach().numpy()
            i += l
    return feats


def mk_sims(df, nm_mdl=None):
    if nm_mdl is None:
        nm_mdl = "pvl/labse_bert"
    feats = mk_feats(df, nm_mdl)
    sims = cosine_similarity(feats)
    return sims
