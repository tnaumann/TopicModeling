function plotBullsEye(Z, bestcv, bestc, bestg, bestSens, bestSpec, c_begin, c_end, g_begin, g_end)

    xlin = linspace(c_begin,c_end,size(Z,1));
    ylin = linspace(g_begin,g_end,size(Z,2));
    [X,Y] = meshgrid(xlin,ylin); 
    Z = Z';
    acc_range = (ceil(bestcv)-3.5:.5:ceil(bestcv));
    [C,hC] = contour(X,Y,Z,acc_range);

    %legend plot
    set(get(get(hC,'Annotation'),'LegendInformation'),'IconDisplayStyle','Children')
    ch = get(hC,'Children');
    tmp = cell2mat(get(ch,'UserData'));
    [M,N] = unique(tmp);
    c = setxor(N,1:length(tmp));
    for i = 1:length(N)
        set(ch(N(i)),'DisplayName',num2str(acc_range(i)))
    end  
    for i = 1:length(c) 
        set(get(get(ch(c(i)),'Annotation'),'LegendInformation'),'IconDisplayStyle','Off')
    end
    legend('show')  

    %bullseye plot
    hold on;
    plot(log2(bestc),log2(bestg),'o','Color',[0 0.5 0],'LineWidth',2,'MarkerSize',15); 
    axs = get(gca);
    plot([axs.XLim(1) axs.XLim(2)],[log2(bestg) log2(bestg)],'Color',[0 0.5 0],'LineStyle',':')
    plot([log2(bestc) log2(bestc)],[axs.YLim(1) axs.YLim(2)],'Color',[0 0.5 0],'LineStyle',':')
    hold off;
    title({['Best log2(C) = ', num2str(bestc), ',  log2(gamma) = ',num2str(bestg),',  Accuracy = ',num2str(bestcv),'%']; ...
        ['Sensitivity = ', num2str(bestSens), ', Specificity = ', num2str(bestSpec)]; ...
        ['(C = ',num2str(2^(bestc)),',  gamma = ',num2str(2^(bestg)),')']})
    xlabel('log2(C)')
    ylabel('log2(gamma)')