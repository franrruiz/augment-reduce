function [yTr yTs K idxTsRm] = keep_first_label(xTr, xTs, flag_mexFile)

if(flag_mexFile)
    [Ntr Kini] = size(xTr);
    Nts = size(xTs,1);
    [yTr yTs] = keep_first_label_c(xTr, xTs, Ntr, Nts, Kini);
    K = length(unique(yTr));
    idxTsRm = find(yTs<=0);
else
    Ntr = size(xTr,1);
    Nts = size(xTs,1);
    yTr = zeros(Ntr,1);
    yTs = zeros(Nts,1);

    idxTsRm = [];
    lbl_set = [];
    count = 1;
    flag0 = 0;
    for nn=1:Ntr
         idxN0 = find(xTr(nn,:)>0,1);
         if(~isempty(idxN0))
             kk = find(idxN0==lbl_set);
             if(isempty(kk))
                 lbl_set = [lbl_set idxN0];
                 yTr(nn) = count;
                 count = count+1;
             else
                 yTr(nn) = kk;
             end
         else
             flag0 = 1;
             yTr(nn) = 0;
         end
    end
    yTr = yTr+flag0;
    K = flag0+length(lbl_set);

    for nn=1:Nts
        idxN0 = find(xTs(nn,:)>0,1);
        if(~isempty(idxN0))
            kk = find(idxN0==lbl_set);
            if(isempty(kk))
                idxTsRm = [idxTsRm; nn];
                yTs(nn) = -1;
            else
                yTs(nn) = kk;
            end
        else
            yTs(nn) = 0;
        end
    end
    yTs = yTs+flag0;
end
