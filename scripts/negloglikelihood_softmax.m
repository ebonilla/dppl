function negloglike = negloglikelihood_softmax(all_w,x,y,nstates, ...
					    weights,ptr_func,ptr_gradfunc, lambda_w)
 
negloglike = - loglikelihood_softmax(all_w,x,y,nstates, ...
				     weights,ptr_func,ptr_gradfunc, lambda_w);