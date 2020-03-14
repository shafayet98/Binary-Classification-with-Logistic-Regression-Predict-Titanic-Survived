function probability_modify = probMod(prob)

    for i = 1: length(prob),
        if prob(i) >= 0.5
            probability_modify(i) = 1;
        elseif prob(i) < 0.5
            probability_modify(i) = 0;
        endif;
    endfor;

end;