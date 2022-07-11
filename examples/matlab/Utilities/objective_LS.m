function L = objective_LS(map,X,Y)

eval = map.Evaluate(X);
L = mean((eval-Y).^2);
end