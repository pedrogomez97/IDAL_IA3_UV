%% test data
%%load the test data

test2 = imageDatastore('test',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
inputSize=[224,224,3];
testValidation2 = augmentedImageDatastore(inputSize(1:2),test2);
%% please load trained network
[YPredTest2,probs2]= classify(trainedNet,testValidation2);

%%test accuracy 

accuracyTest2 = mean(YPredTest2 == test2.Labels);
display(accuracyTest2)

%% Plot the confusion matrix.

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm2 = confusionchart(test2.Labels,YPredTest2);
cm2.Title = 'Confusion Matrix for Validation Data';
cm2.ColumnSummary = 'column-normalized';
cm2.RowSummary = 'row-normalized';

%% ROC for CNN

a= double(test2.Labels);
b = probs2;
figure
[X,Y] = perfcurve(a,b(:,1),1);
plot(X,Y,'linewidth',2);
grid
xlabel('1- Specificity')
ylabel('Sensitivity')
title('ROC for Classification CNN')
 