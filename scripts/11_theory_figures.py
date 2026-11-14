"""Figures pédagogiques du chapitre théorique (BM25, RRF, nDCG)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
G="#009B7A"; GR="#7A7F85"; D="#2B2F33"; A="#E4572E"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,3,figsize=(11,3.1))
tf=np.linspace(0,20,200)
for k1,c in [(0.5,A),(1.2,G),(3.0,D)]:
    ax[0].plot(tf,tf*(k1+1)/(tf+k1),color=c,label=f"$k_1={k1}$")
ax[0].plot(tf,tf,"--",color=GR,lw=1,label="TF brut")
ax[0].set_ylim(0,6); ax[0].set_xlabel("fréquence du terme tf"); ax[0].set_ylabel("contribution de tf")
ax[0].set_title("(a) Saturation de tf",fontsize=10); ax[0].legend(frameon=False,fontsize=8)
r=np.linspace(0.25,4,200)  # |d|/avgdl
for b,c in [(0,A),(0.5,GR),(0.75,G),(1,D)]:
    tfn=2*(1.2+1)/(2+1.2*(1-b+b*r)); ax[1].plot(r,tfn,color=c,label=f"$b={b}$")
ax[1].set_xlabel("longueur relative $|d|/\\mathrm{avgdl}$"); ax[1].set_ylabel("contribution (tf = 2)")
ax[1].set_title("(b) Normalisation de longueur",fontsize=10); ax[1].legend(frameon=False,fontsize=8)
rk=np.arange(1,31)
for k,c in [(1,A),(10,GR),(60,G)]:
    ax[2].plot(rk,1/(k+rk),"o-",ms=3,color=c,label=f"$k={k}$")
ax[2].set_xlabel("rang $r$"); ax[2].set_ylabel("$1/(k+r)$")
ax[2].set_title("(c) Contribution RRF",fontsize=10); ax[2].legend(frameon=False,fontsize=8)
plt.tight_layout(); plt.savefig(str(__import__("pathlib").Path(__file__).resolve().parents[1]/"report"/"figures")+"/theory_bm25_rrf.png",dpi=200); plt.close()
# nDCG: décote logarithmique
fig,ax=plt.subplots(figsize=(4.2,2.8))
rk=np.arange(1,11); ax.bar(rk,1/np.log2(rk+1),color=G)
ax.plot(rk,1/rk,"o-",color=A,label="$1/r$ (MRR)")
ax.set_xlabel("rang $r$"); ax.set_ylabel("poids de la position")
ax.legend(["$1/r$ (rang réciproque)"],frameon=False); ax.set_xticks(rk)
ax.set_title("Barres : $1/\\log_2(r+1)$ (nDCG)",fontsize=9)
plt.tight_layout(); plt.savefig(str(__import__("pathlib").Path(__file__).resolve().parents[1]/"report"/"figures")+"/theory_discount.png",dpi=200); plt.close()
