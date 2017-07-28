function [w, b] = primal_softmargin(X, y, beta)

 [~, n] = size(X);
 [t, ~] = size(y);
 
 H = eye(n);
 H = [beta*H zeros(n,1+t);
            zeros(t+1, n+t+1)];
 
 A = -1*([zeros(t,n+1) eye(t,t)]+diag(y)*[X zeros(t,t+1)]+[zeros(t,n) y zeros(t,t)]);
  
 A = [A;
       zeros(t,n+1) -1*eye(t,t)];
   
 f = [zeros(n+1, 1);
      ones(t,1)];
  
 q = quadprog(H,f,A,-1*[ones(t,1); zeros(t,1)]);
       
 w = q(1:n, 1);
 b = q(n+1, 1);
 
 
end