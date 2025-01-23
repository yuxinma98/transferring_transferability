function [] = generate_dataset()
    clear; %clear memory

    for i = 1:4
        task = ['generate_task', num2str(i), '_dataset'];
        generator = str2func(task);
        out_directory = ['data/task', num2str(i), '/']; %['./data/task', num2str(i), '/']; % 
        disp(task);
        disp(out_directory);

        %Generate train dataset
        for N = 500:500:3000 %number of points per set
            disp(N);
            train_file = [out_directory, 'train_', int2str(N), '.mat'];
            val_file = [out_directory, 'val_', int2str(N), '.mat'];
            test_file = [out_directory, 'test_', int2str(N), '.mat'];

            if ~isfile(train_file)
                save_dataset(2^10, N, train_file, generator)
            end
            if ~isfile(val_file)
                save_dataset(512, N, val_file, generator)
            end
            if ~isfile(test_file)
                save_dataset(512, N, test_file, generator)
            end
        end

        %Generate truth for plotting
        truth_file = [out_directory, 'truth.mat'];
        if ~isfile(truth_file)
            save_dataset(2^14, 1, truth_file, generator)
        end

        %additional test set with different number of points per sets
        for N = 3000:500:5000 %number of train distributions
            disp(N);
            test_file = [out_directory, 'test_', int2str(N), '.mat'];
            if ~isfile(test_file)
                save_dataset(512, N, test_file, generator)
            end
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