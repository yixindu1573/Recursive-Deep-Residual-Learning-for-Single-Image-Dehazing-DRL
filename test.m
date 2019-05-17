%%% test the model performance
clear; clc;close all;
run( '/libs/vlfeat-0.9.20/toolbox/vl_setup');

dir_matConvNet='/libs/matconvnet/matlab/';
addpath(genpath(dir_matConvNet));
run([dir_matConvNet 'vl_setupnn.m']);

mkdir('result');

format compact;

addpath(fullfile('utilities'));

showResult  = 1;
useGPU      = 1;
pauseTime   = 0;

modelName   = 'model.mat';

files = dir('testData');
for i = 3:length(files)
    input = im2single(imread(fullfile('testData',files(i).name)));
    load(modelName);
    net = vl_simplenn_tidy(net);
    net.layers = net.layers(1:end-1);
    net = vl_simplenn_tidy(net);

    %%% move to gpu
    if useGPU
        net = vl_simplenn_move(net, 'gpu') ;
    end

    % first iteration
    %%% convert to GPU
    if useGPU
        input1 = gpuArray(input);
    end

    res1    = vl_simplenn(net,input1,[],[],'conserveMemory',true,'mode','test');
    output1 = input1 - res1(end).x;

    %%% convert to CPU
    if useGPU
        output1 = gather(output1);
        input1  = gather(input1);
    end

    %second iteration
    if useGPU
        input2 = gpuArray(output1);
    end
    res2    = vl_simplenn(net,input2,[],[],'conserveMemory',true,'mode','test');
    output2 = input2 - res2(end).x;

    %%% convert to CPU
    if useGPU
        output2 = gather(output2);
        input2  = gather(input2);
    end

    %third iteration
    if useGPU
        input3 = gpuArray(output2);
    end
    res3    = vl_simplenn(net,input3,[],[],'conserveMemory',true,'mode','test');
    output3 = input3 - res3(end).x;

    %%% convert to CPU
    if useGPU
        output3 = gather(output3);
        input3  = gather(input3);
    end

    result = [input output1 output2 output3];
    figure(2);imshow(result);
    imwrite(result, fullfile('result',strcat(files(i).name(1:2),'.png')));

end




