"""EXPERIMENTAL ONLY. Full-batch algebra; no formal runner imports this file."""
import numpy as np
from scipy.linalg import inv
def loading_numerator(zeroed,latents,weights):
 return zeroed@(latents.conj()*weights[None,:]).T
def latent_moment(latents,weights,base):
 return weights.sum()*base+latents@(latents.conj()*weights[None,:]).T
def real_quadratic(X1,X2):
 return np.einsum('dn,dn->n',X1.real,X2.real,optimize=False)+np.einsum('dn,dn->n',X1.imag,X2.imag,optimize=False)
def residual_moment(zeroed,old_lambdalatents,weights):
 # The OLD loadings-times-latents are essential to the author's update order.
 return (np.einsum('dn,dn,n->d',zeroed.real-old_lambdalatents.real,zeroed.real,weights,optimize=False)+
         np.einsum('dn,dn,n->d',zeroed.imag-old_lambdalatents.imag,zeroed.imag,weights,optimize=False))/weights.sum()
def log_gaussian_no_det(self,k,data):
 X1=(data-self._means[k]).T
 X2=self._inv_covs[k]@X1
 return -np.log(np.pi)*data.shape[1]-real_quadratic(X1,X2)
def em_step(self,data,PPCA):
 self.logLs,self.rs=self._calc_probs(data)
 sumrs=np.sum(self.rs,axis=1)
 betas=np.transpose(self._lambdas.conj(),[0,2,1])@self._inv_covs
 for k in range(self.n_components):
  latents=betas[k]@(data.T-self._means[k,:,None])
  latent_base=np.eye(self.M)-betas[k]@self._lambdas[k]
  old_lambdalatents=self._lambdas[k]@latents
  self._means[k]=np.sum(self.rs[k]*(data.T-old_lambdalatents),axis=1)/sumrs[k]
  zeroed=data.T-self._means[k,:,None]
  cross=loading_numerator(zeroed,latents,self.rs[k])
  gamma=latent_moment(latents,self.rs[k],latent_base)
  self._lambdas[k]=np.dot(cross,inv(gamma))
  psis=residual_moment(zeroed,old_lambdalatents,self.rs[k])
  self._psis[k]=np.clip(psis,1e-6,np.inf)
  if PPCA:self._psis[k]=np.mean(self._psis[k])*np.ones(self.D)
  self.amps[k]=sumrs[k]/data.shape[0]
  del latents,latent_base,old_lambdalatents,zeroed,cross,gamma,psis
 if self.lock_psis:
  psi=np.dot(sumrs,self._psis)/np.sum(sumrs)
  self._psis=np.full_like(self._psis,psi)
 del betas
 self._update_covs()
