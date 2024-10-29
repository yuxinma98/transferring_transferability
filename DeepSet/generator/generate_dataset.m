function [] = generate_dataset()
    clear; %clear memory

    for i = 1:4
        task = ['generate_task', num2str(i), '_dataset'];
        generator = str2func(task);
        out_directory = ['data/task', num2str(i), '/'] %['./data/task', num2str(i), '/']; % 
        disp(task);
        disp(out_directory);

        L = 2^10; %Number of sets
        N = 500; %Number of points per set

        %Generate train dataset
        save_dataset(L, N, [out_directory,'train.mat'], generator)

        L=512; %Number of sets for test and validation
        %Generate test dataset
        save_dataset(L, N, [out_directory,'test.mat'], generator)
        %Generate validation dataset
        save_dataset(L, N, [out_directory,'val.mat'], generator)

        %Generate truth for plotting
        save_dataset(2^14, 1, [out_directory,'truth.mat'], generator)

        %additional test set with different number of points per sets
        L = 512;
        for N = 1000:500:5000 %number of train distributions
            disp(N);
            save_dataset(L, N, [out_directory, 'data_', int2str(N), '.mat'], generator)
        end
        display(['Data generated for task' num2str(i)])
    end
    display('Done')
end


function [] = save_dataset(L, N, fname, generator)
    [X, Y, X_parameter] = generator(L, N);
    X = cell2mat(X);
    save(fname, '-v7.3')
end
