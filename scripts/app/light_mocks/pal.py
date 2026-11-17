P = {
 "A-tufte": dict(bg="#fffff8", felt="#fbfaf0", zone="#f2f0e4", sheet="#f8f6eb", text="#111111", muted="#5c584f", faint="#c8c3b3",
                 you="#3a5a8a", bot="#a33a2e", accent="#7d5f1c", danger="#a33a2e", neg="#b4554a", i1="#dfe6f0", i2="#ecf0f6"),
 "B-solarized": dict(bg="#fdf6e3", felt="#f9f1dc", zone="#eee8d5", sheet="#f6efdc", text="#073642", muted="#52676e", faint="#b9bdb4",
                 you="#1b64a0", bot="#b8322a", accent="#7a5c00", danger="#b8322a", neg="#c45a4c", i1="#d9e6ee", i2="#e6eef0"),
 "C-washi": dict(bg="#f5f1e8", felt="#f1ece1", zone="#e8e2d4", sheet="#faf7f0", text="#2a2721", muted="#666054", faint="#c4bcaa",
                 you="#46628c", bot="#a5483c", accent="#745a26", danger="#a1453a", neg="#b0594d", i1="#e2e6ee", i2="#e9ebef"),
}
def lum(h):
    c=[int(h[i:i+2],16)/255 for i in (1,3,5)]
    c=[x/12.92 if x<=0.03928 else ((x+0.055)/1.055)**2.4 for x in c]
    return 0.2126*c[0]+0.7152*c[1]+0.0722*c[2]
def cr(a,b):
    la,lb=sorted([lum(a),lum(b)],reverse=True); return (la+0.05)/(lb+0.05)
if __name__=="__main__":
    for k,p in P.items():
        worst={f:min(cr(p[f],p[s]) for s in ("bg","felt","zone","sheet","i2")) for f in ("text","muted","accent","danger","you","bot")}
        print(k, {f:round(v,2) for f,v in worst.items()})
