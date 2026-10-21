from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor
for n,(M,P) in {"openai/clip-vit-large-patch14":(CLIPModel,CLIPProcessor),"google/siglip-so400m-patch14-384":(AutoModel,AutoProcessor)}.items():
    try: M.from_pretrained(n); P.from_pretrained(n); print("ok",n)
    except Exception as e: print("FAIL",n,repr(e)[:300])
