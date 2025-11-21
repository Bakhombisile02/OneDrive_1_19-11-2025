### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 1ca7d017-da7f-4d9d-9dd3-ebe327bdf140
begin
    using CSV, DataFrames
    using MultivariateStats
    using ManifoldLearning
    using EvoTrees
    using TSne
    using MLJBase
    using Plots
    using Printf
    using StatsBase
    using MLJLIBSVMInterface, LIBSVM
    using MLJDecisionTreeInterface
    using NearestNeighborModels
    using MLJ
    using Random
    using LinearAlgebra
	using StatsPlots
	using Statistics
    
    # Load necessary MLJ models
    @load ProbabilisticSVC pkg=LIBSVM
using LIBSVM # This is needed to access LIBSVM.Kernel
    EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
end


# ╔═╡ 23550ef3-a47e-4d8d-952f-e63bc56b6cf8
md"""### Task 1"""

# ╔═╡ a32f2f31-42df-4410-86df-3ad3784fcf7e
begin
	df = CSV.read("Parkinsons_Speech-Features.csv", DataFrame)
	
	println(size(df))
	first(df, 5)
end

# ╔═╡ 08de05af-d020-46c3-96ee-8fcdc8557466
begin
	schemes = Dict(
	    "Baseline" => 3:23,
	    "IntensityFormantBandwidth" => 24:34,
	    "VocalFold" => 35:56,
	    "MFCC" => 57:140,
	    "Wavelet" => 141:322,
	    "TQWT" => 323:754
	)
	
	# Class label
	y = convert(Vector{Int}, df[:, 755])   # 1 = PD, 0 = HC
	subject_id = df[:, 1]   
end

# ╔═╡ e8691046-2118-4f57-bab3-17bed354e713
begin
	function standardize(X::DataFrame)
	    numeric = Matrix(X)
	    μ = mean(numeric, dims=1)
	    σ = std(numeric, dims=1)
	    return (numeric .- μ) ./ σ
	end
	
	X_schemes = Dict(name => standardize(df[:, cols]) for (name, cols) in schemes)
end

# ╔═╡ 01efd005-143a-4618-b7fd-68c3652389e4
function visualize_scheme(X, y, scheme_name;
                          pca_dim::Int=50,
                          n_neighbors::Int=15,
                          perplexity::Int=30)

    n_features, n_samples = size(X)
    pca_dim = min(pca_dim, n_features, n_samples - 1)
    
    pca_model = MultivariateStats.fit(PCA, X; maxoutdim=pca_dim)
    X_pca_reduced = MultivariateStats.transform(pca_model, X)
    
    iso_model = ManifoldLearning.fit(ManifoldLearning.Isomap, X_pca_reduced; 
                                     k=n_neighbors, maxoutdim=2)
    X_iso = ManifoldLearning.predict(iso_model)
    
    X_tsne = tsne(X_pca_reduced', 2, pca_dim, perplexity)
    
    colors = [y[i] == 0 ? :blue : :red for i in 1:length(y)]
    
    p1 = scatter(X_iso[1,:], X_iso[2,:], 
                 color=colors,
                 title="$scheme_name – Isomap",
                 xlabel="Component 1", ylabel="Component 2", 
                 label="",
                 markersize=3, alpha=0.6,
                 markerstrokewidth=0)

    p2 = scatter(X_tsne[:,1], X_tsne[:,2], 
                 color=colors,
                 title="$scheme_name – t-SNE",
                 xlabel="Dim 1", ylabel="Dim 2", 
                 label="",
                 markersize=3, alpha=0.6,
                 markerstrokewidth=0)

    scatter!(p1, [], [], color=:blue, label="Healthy Control", 
             markersize=5, legend=:topright)
    scatter!(p1, [], [], color=:red, label="Parkinson's Disease", markersize=5)
    
    scatter!(p2, [], [], color=:blue, label="Healthy Control", 
             markersize=5, legend=:topright)
    scatter!(p2, [], [], color=:red, label="Parkinson's Disease", markersize=5)

    plot(p1, p2, layout=(1,2), size=(1000,400))
end

# ╔═╡ 95224368-0653-4c50-a31c-22bfc2d3eee7
begin
	plots_list = []
	for (name, X) in X_schemes
	    p = visualize_scheme(X', y, name)
	    push!(plots_list, p)
	end
	
	# Display all at once
	plot(plots_list..., layout=(6,1), size=(1000, 2400))
end

# ╔═╡ 0e1503e1-89e6-4638-be96-268c540d6b8c
md"""#### Separability"""

# ╔═╡ e1b90593-00f0-4984-b0c0-f2ea78a8fedb
function compute_fisher_ratio(X, y)
  
    X_hc = X[:, y .== 0]
    X_pd = X[:, y .== 1]
    
    μ_hc = mean(X_hc, dims=2)
    μ_pd = mean(X_pd, dims=2)
    μ_all = mean(X, dims=2)
    
    n_hc = size(X_hc, 2)
    n_pd = size(X_pd, 2)
    S_b = n_hc * (μ_hc .- μ_all) * (μ_hc .- μ_all)' + 
          n_pd * (μ_pd .- μ_all) * (μ_pd .- μ_all)'
    
    S_w_hc = cov(X_hc', corrected=true)
    S_w_pd = cov(X_pd', corrected=true)
    S_w = (n_hc * S_w_hc + n_pd * S_w_pd) / (n_hc + n_pd)
    
    fisher_ratio = tr(S_b) / tr(S_w)
    
    return fisher_ratio
end

# ╔═╡ af3237ee-28ea-43e4-b9fe-67cf4ad97d27
function compute_separability(X_schemes, y; save_plot=false)
 
    results = DataFrame(
        Scheme = String[],
        N_Features = Int[],
        Fisher_Ratio = Float64[]
    )
    
    println("=" ^ 70)
    println("COMPUTING SEPARABILITY METRICS")
    println("=" ^ 70)
    println()
    
    for (name, X) in X_schemes
        n_features = size(X, 2)
        X_t = X'  # Transpose to features × samples
        
        fisher = compute_fisher_ratio(X_t, y)
        
        push!(results, (name, n_features, fisher))
        
        println("$name:")
        println("  Features: $n_features")
        println("  Fisher Ratio: $(round(fisher, digits=4))")
        println()
    end
    
    sort!(results, :Fisher_Ratio, rev=true)
    
    println("=" ^ 70)
    println("RANKED BY SEPARABILITY (Higher = Better)")
    println("=" ^ 70)
    println(results)
    println()
    
    p = bar(results.Scheme, results.Fisher_Ratio,
            title="PD vs HC Separability by Feature Scheme",
            xlabel="Feature Scheme", 
            ylabel="Fisher Ratio (higher = better)",
            legend=false, xrotation=45, color=:steelblue,
            size=(800,500))
    display(p)
    
    if save_plot
        savefig(p, "separability_comparison.png")
        println("Saved: separability_comparison.png")
    end
    
    return results
end

# ╔═╡ 99256a50-18d6-4e5c-a742-866a55ba0060
results = compute_separability(X_schemes, y, save_plot=true)

# ╔═╡ b40fcb46-1101-4ecf-aaec-ff90686e12ca
md"""### Task 2"""

# ╔═╡ a657d8ca-fde8-4b87-a4da-3b3ae96c04ed
import MLJ:fit!, predict, machine

# ╔═╡ 5f72ca76-2e44-49f3-83b9-6cae21a8e690
begin
    DTree = @load DecisionTreeClassifier pkg=DecisionTree
    SVC = @load SVC pkg=LIBSVM
    KNN = @load KNNClassifier pkg = NearestNeighborModels
end

# ╔═╡ f8dcdd9a-9318-469c-b182-7a87c5cf816e
function create_subject_folds(subject_id, n_folds=5; seed=42)
   
    Random.seed!(seed)
    
    # Get unique subjects
    unique_subjects = unique(subject_id)
    n_subjects = length(unique_subjects)
    
    # Shuffle subjects
    shuffled_subjects = shuffle(unique_subjects)
    
    # Assign subjects to folds
    subject_to_fold = Dict()
    for (i, subj) in enumerate(shuffled_subjects)
        fold = mod1(i, n_folds)
        subject_to_fold[subj] = fold
    end
    
    # Map each sample to its fold based on subject
    fold_indices = [subject_to_fold[subj] for subj in subject_id]
    
    return fold_indices
end

# ╔═╡ afaf5667-9337-4837-aa33-f3886a7102ea
function compute_auc(y_true, y_prob)
  
    sorted_idx = sortperm(y_prob, rev=true)
    y_sorted = y_true[sorted_idx]
    
    n_pos = sum(y_true .== 1)
    n_neg = sum(y_true .== 0)
    
    if n_pos == 0 || n_neg == 0
        return NaN
    end
    
    tp = 0
    fp = 0
    auc = 0.0
    
    for i in 1:length(y_sorted)
        if y_sorted[i] == 1
            tp += 1
        else
            fp += 1
            auc += tp
        end
    end
    
    return auc / (n_pos * n_neg)
end

# ╔═╡ 8771a550-2183-4918-b191-11b97fb3e6b2
function compute_metrics(y_true, y_pred, y_prob_param=nothing)
   
    tp = sum((y_true .== 1) .& (y_pred .== 1))
    tn = sum((y_true .== 0) .& (y_pred .== 0))
    fp = sum((y_true .== 0) .& (y_pred .== 1))
    fn = sum((y_true .== 1) .& (y_pred .== 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp > 0 ? tp / (tp + fn) : 0.0
    specificity = tn > 0 ? tn / (tn + fp) : 0.0
    precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
    f1 = (precision + sensitivity) > 0 ? 2 * (precision * sensitivity) / (precision + sensitivity) : 0.0
    
    auc_val = nothing
    if y_prob_param !== nothing
        auc_val = compute_auc(y_true, y_prob_param)
    end
    
    return Dict(
        "Accuracy" => accuracy,
        "Sensitivity" => sensitivity,
        "Specificity" => specificity,
        "Precision" => precision,
        "F1" => f1,
        "AUC" => auc_val
    )
end

# ╔═╡ 19363da8-f0f9-410c-84a0-ad58321b4739
function cross_validate_model(X, y, subject_id, model_constructor; n_folds=5, seed=42)
   
    fold_indices = create_subject_folds(subject_id, n_folds; seed=seed)
    
    fold_results = []
    
    for fold in 1:n_folds
        # Split data
        test_mask = fold_indices .== fold
        train_mask = .!test_mask
        
        X_train = X[train_mask, :]
        y_train = y[train_mask]
        X_test = X[test_mask, :]
        y_test = y[test_mask]
        
        model = model_constructor()
        
        mach = machine(model, X_train, categorical(y_train))
        fit!(mach, verbosity=0)
        
        y_pred_raw = predict(mach, X_test)
        
        if typeof(model) <: SVC
            y_pred_mode = y_pred_raw
            y_prob_vals = nothing  
        else
            y_pred_mode = mode.(y_pred_raw)
            try
                y_prob_vals = pdf.(y_pred_raw, 1) 
            catch
                y_prob_vals = nothing
            end
        end
        
        metrics = compute_metrics(y_test, y_pred_mode, y_prob_vals)
        push!(fold_results, metrics)
    end
    
    metric_names = keys(fold_results[1])
    aggregated = Dict()
    
    for metric in metric_names
        values = [fr[metric] for fr in fold_results if fr[metric] !== nothing]
        if !isempty(values)
            aggregated[metric * "_mean"] = mean(values)
            aggregated[metric * "_std"] = std(values)
        else
            aggregated[metric * "_mean"] = NaN
            aggregated[metric * "_std"] = NaN
        end
    end
    
    return aggregated
end


# ╔═╡ e2944b0f-7cd0-4356-ae72-c0ab63a61301
function evaluate_all_schemes(X_schemes, y, subject_id; n_folds=5, seed=42)
    
    models = Dict(
        "Decision Tree" => () -> DTree(max_depth=10),
        "SVM" => () -> SVC(),
        "KNN" => () -> KNN(K=5)
    )
    
    results = DataFrame(
        Scheme = String[],
        Model = String[],
        N_Features = Int[],
        Accuracy = Float64[],
        Accuracy_Std = Float64[],
        Sensitivity = Float64[],
        Sensitivity_Std = Float64[],
        Specificity = Float64[],
        Specificity_Std = Float64[],
        F1 = Float64[],
        F1_Std = Float64[],
        AUC = Float64[],
        AUC_Std = Float64[]
    )
    
    println("=" ^ 80)
    println("CROSS-VALIDATION CLASSIFICATION EVALUATION")
    println("Folds: $n_folds (Subject-based)")
    println("=" ^ 80)
    println()
    
    for (scheme_name, X) in X_schemes
        n_features = size(X, 2)
        X_df = DataFrame(X, :auto)
        
        println("Evaluating: $scheme_name ($n_features features)")
        println("-" ^ 60)
        
        for (model_name, model_constructor) in models
            print("  $model_name... ")
            flush(stdout)
            
            try
                cv_results = cross_validate_model(X_df, y, subject_id, model_constructor; 
                                                 n_folds=n_folds, seed=seed)

                push!(results, (
                    scheme_name,
                    model_name,
                    n_features,
                    cv_results["Accuracy_mean"],
                    cv_results["Accuracy_std"],
                    cv_results["Sensitivity_mean"],
                    cv_results["Sensitivity_std"],
                    cv_results["Specificity_mean"],
                    cv_results["Specificity_std"],
                    cv_results["F1_mean"],
                    cv_results["F1_std"],
                    cv_results["AUC_mean"],
                    cv_results["AUC_std"]
                ))
                
                println("Acc: $(round(cv_results["Accuracy_mean"]*100, digits=2))% ± $(round(cv_results["Accuracy_std"]*100, digits=2))%")
            catch e
                println("ERROR: $e")
                push!(results, (
                    scheme_name, model_name, n_features,
                    NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN
                ))
            end
        end
        println()
    end
    
    return results
end

# ╔═╡ 86cb37a0-e006-4b74-8ccb-99f149d78fe1
function plot_comparison_by_metric(results, metric="Accuracy"; save_plot=false)
  
    schemes = unique(results.Scheme)
    models = unique(results.Model)
    
    data = zeros(length(schemes), length(models))
    errors = zeros(length(schemes), length(models))
    
    for (i, scheme) in enumerate(schemes)
        for (j, model) in enumerate(models)
            row = results[(results.Scheme .== scheme) .& (results.Model .== model), :]
            if nrow(row) > 0
                data[i, j] = row[1, Symbol(metric)]
                errors[i, j] = row[1, Symbol(metric * "_Std")]
            end
        end
    end
    
    p = groupedbar(schemes, data * 100,
                   bar_position=:dodge,
                   labels=reshape(models, 1, :),
                   title="$metric Comparison Across Feature Schemes",
                   xlabel="Feature Scheme",
                   ylabel="$metric (%)",
                   xrotation=45,
                   legend=:topright,
                   size=(1000, 600),
                   ylims=(0, 100))
    
    display(p)
    
    if save_plot
        savefig(p, "$(lowercase(metric))_comparison.png")
        println("Saved: $(lowercase(metric))_comparison.png")
    end
    
    return p
end

# ╔═╡ 5bb112bf-6f74-403b-bef3-2251224d6b81
function plot_all_metrics(results; save_plot=false)
   
    metrics = ["Accuracy", "Sensitivity", "Specificity", "F1"]
    plots = []
    
    for metric in metrics
        p = plot_comparison_by_metric(results, metric; save_plot=false)
        push!(plots, p)
    end
    
    combined = plot(plots..., layout=(2, 2), size=(1600, 1200))
    display(combined)
    
    if save_plot
        savefig(combined, "all_metrics_comparison.png")
        println("Saved: all_metrics_comparison.png")
    end
    
    return combined
end

# ╔═╡ 313afa4b-4a9c-42c8-a1aa-eb66ea6462d7
function display_summary_table(results)
    
    println("\n" * "=" ^ 80)
    println("SUMMARY: Best Results by Scheme")
    println("=" ^ 80)
    
    summary = DataFrame(
        Scheme = String[],
        Best_Model = String[],
        Accuracy = String[],
        Sensitivity = String[],
        Specificity = String[],
        F1 = String[]
    )
    
    for scheme in unique(results.Scheme)
        scheme_results = results[results.Scheme .== scheme, :]
        best_idx = argmax(scheme_results.Accuracy)
        best = scheme_results[best_idx, :]
        
        push!(summary, (
            scheme,
            best.Model,
            "$(round(best.Accuracy*100, digits=2))% ± $(round(best.Accuracy_Std*100, digits=2))%",
            "$(round(best.Sensitivity*100, digits=2))% ± $(round(best.Sensitivity_Std*100, digits=2))%",
            "$(round(best.Specificity*100, digits=2))% ± $(round(best.Specificity_Std*100, digits=2))%",
            "$(round(best.F1*100, digits=2))% ± $(round(best.F1_Std*100, digits=2))%"
        ))
    end
    
    println(summary)
    println()
    
    best_overall_idx = argmax(results.Accuracy)
    best_overall = results[best_overall_idx, :]
    
    println("OVERALL BEST CONFIGURATION:")
    println("  Scheme: $(best_overall.Scheme)")
    println("  Model: $(best_overall.Model)")
    println("  Accuracy: $(round(best_overall.Accuracy*100, digits=2))% ± $(round(best_overall.Accuracy_Std*100, digits=2))%")
    println("  Sensitivity: $(round(best_overall.Sensitivity*100, digits=2))% ± $(round(best_overall.Sensitivity_Std*100, digits=2))%")
    println("  Specificity: $(round(best_overall.Specificity*100, digits=2))% ± $(round(best_overall.Specificity_Std*100, digits=2))%")
    println("  F1 Score: $(round(best_overall.F1*100, digits=2))% ± $(round(best_overall.F1_Std*100, digits=2))%")
    println("=" ^ 80)
    
    return summary
end

# ╔═╡ 2eb18ebe-5a47-4ad8-a7d4-44ef58e9f657
cv_results = evaluate_all_schemes(X_schemes, y, subject_id; n_folds=5, seed=42)

# ╔═╡ f15b56a3-a334-4dd3-beee-34e08e4994d6
summary = display_summary_table(cv_results)

# ╔═╡ cf3c8858-f6ff-4c11-9869-0dd04c34cf25
plot_all_metrics(cv_results, save_plot=true)

# ╔═╡ b531fda7-c102-4ee0-8214-5f523caa2ffe
md"""
## Task 3 & 4 — mRMR + GBDT & SVM (AUPRC/MCC, threshold tuning, stability, gender)
We select features with **mRMR** (inside each training fold), train **GBDT (EvoTrees)** and **SVM (RBF)**, optimize by **AUPRC** (tie-break MCC), tune the **probability threshold** to **maximize MCC**, show **PR curves** and **threshold-vs-MCC**, report **outer-CV mean±CI**, check **feature stability** (Jaccard), and repeat on **gender-specific** subsets.
"""


# ╔═╡ d332c4a3-4e1d-4b2f-bcc6-977df4555181
# --- Part 1: Model Definitions, Metrics, and Feature Selection Helpers ---
begin
    # Define model constructors for clarity and convenience
    GBDT(; η=0.05, depth=5, rounds=300, subs=0.8, col=0.8) =
        EvoTreeClassifier(loss=:logit, eta=η, max_depth=depth,
                              nrounds=rounds, subsample=subs, colsample=col)

    # This now correctly uses ProbabilisticSVC for probability outputs and the proper kernel syntax.
    function SVM_RBF(; C::Float64=1.0, γ::Float64=0.1)
        return ProbabilisticSVC(
            kernel    = LIBSVM.Kernel.RadialBasis,
            gamma     = γ,
            cost      = C,
            tolerance = 1e-3,
            shrinking = true
        )
    end

    # This function handles the different weighting schemes for SVM and GBDT.
    function train_predict_scores!(model, Xtr::Matrix{Float64}, ytr::Vector{Int}, Xva::Matrix{Float64}; weights=nothing)
        Xtr_df = DataFrame(Xtr, :auto)
        Xva_df = DataFrame(Xva, :auto)
        yc = categorical(ytr)
        
        mach = machine(model, Xtr_df, yc)

        try
            if model isa ProbabilisticSVC
                n_pos = sum(ytr .== 1)
                n_neg = length(ytr) - n_pos
                if n_pos > 0 && n_neg > 0
                    class_weights_dict = Dict(0 => 1.0, 1 => n_neg / n_pos)
                    # For SVM, re-bind the machine with a class weight dictionary.
                    mach = machine(model, Xtr_df, yc, class_weights_dict)
                end
                fit!(mach; verbosity=0) # SVC fit! does not take a weights argument.
            else # Assumed GBDT
                if weights !== nothing
                    # For GBDT, bind the machine with the instance weight vector.
                    mach = machine(model, Xtr_df, yc, weights)
                end
                fit!(mach; verbosity=0)
            end
        catch e
             println("!!! WARNING: Model fitting failed. Error: $e")
             # Fallback to a standard machine if weighting fails.
             mach = machine(model, Xtr_df, yc)
             fit!(mach; verbosity=0)
        end
        
        ŷ = predict(mach, Xva_df)
        
        try
            return pdf.(ŷ, 1)
        catch
            return Float64.(mode.(ŷ) .== 1)
        end
    end

    # This is the most idiomatic and correct way to handle this in Julia.
    function confusion_at(y_true::Vector{Int}, scores::Vector{<:Real}, τ::Float64)
        y_pos      = y_true .== 1
        y_pred_pos = scores .>= τ
        
        tp = sum(y_pos .&  y_pred_pos)
        tn = sum(.!y_pos .& .!y_pred_pos)
        fp = sum(.!y_pos .&  y_pred_pos)
        fn = sum(y_pos .&  .!y_pred_pos)
        
        return (tp, tn, fp, fn)
    end

    # Calculate Matthews Correlation Coefficient
    mcc(tp, tn, fp, fn) = begin
        denom = big(tp+fp)*big(tp+fn)*big(tn+fp)*big(tn+fn)
        denom == 0 ? 0.0 : (tp*tn - fp*fn)/sqrt(denom)
    end

    # **FIXED**: The signature now accepts any Real number type for scores.
    # Generate Precision-Recall curve and calculate Area Under the PR Curve (AUPRC)
    function pr_curve(y_true::Vector{Int}, scores::AbstractVector{<:Real})
        idx = sortperm(scores, rev=true)
        y_sorted = y_true[idx]
        P = sum(y_true .== 1)
        
        if P == 0 || P == length(y_true)
            is_positive = Float64(P > 0)
            return ([0.0, 1.0], [is_positive, is_positive], is_positive)
        end
        
        tp = cumsum(y_sorted)
        fp = (1:length(y_sorted)) .- tp
        
        rec = tp ./ P
        prec = tp ./ (tp .+ fp)
        
        # Build a clean step-plot for accurate AUPRC calculation
        rec_clean = [0.0]
        prec_clean = [1.0]
        last_rec = 0.0
        
        for i in 1:length(rec)
            if rec[i] > last_rec
                push!(rec_clean, rec[i])
                push!(prec_clean, prec[i])
                last_rec = rec[i]
            end
        end

        # Calculate area using the trapezoidal rule on the step plot
        auprc = sum((rec_clean[i] - rec_clean[i-1]) * prec_clean[i] for i in 2:length(rec_clean))
        
        return prec_clean, rec_clean, auprc
    end


    # Find the optimal probability threshold that maximizes MCC
    function best_threshold_mcc(y_true::Vector{Int}, scores::Vector{<:Real}; grid=0.0:0.005:1.0)
        (isempty(scores) || isempty(y_true)) && return 0.5, -1.0
        mcc_vals = [mcc(confusion_at(y_true, scores, τ)...) for τ in grid]
        best_idx = argmax(mcc_vals)
        return grid[best_idx], mcc_vals[best_idx]
    end

    # Discretize a vector into quantile-based bins
    qcut(x::AbstractVector{<:Real}, q::Int=5) = begin
        qs = unique(quantile(skipmissing(x), range(0, 1; length=q+1)))
        if length(qs) <= 1; return ones(Int, length(x)); end
        b = zeros(Int, length(x))
        for i in eachindex(x)
            b[i] = searchsortedfirst(qs, x[i]) - 1
        end
        return clamp.(b, 1, length(qs)-1)
    end

    # Discretize a matrix column by column
    discretize_matrix(X::AbstractMatrix{<:Real}; qbins::Int=5) = 
        hcat([qcut(view(X, :, j), qbins) for j in 1:size(X,2)]...)

    # Calculate mutual information between two discrete vectors
    function mi_discrete(x::AbstractVector{<:Integer}, y::AbstractVector{<:Integer})
        (isempty(x) || isempty(y)) && return 0.0
        max_x, max_y = maximum(x), maximum(y)
        (max_x == 0 || max_y == 0) && return 0.0
        
        p_xy = proportions(x, y, (1:max_x, 1:max_y))
        p_x = proportions(x, 1:max_x)
        p_y = proportions(y, 1:max_y)
        
        I = sum(p_xy[i,j] > 0 ? p_xy[i,j] * log2(p_xy[i,j] / (p_x[i] * p_y[j])) : 0 for i in 1:size(p_xy,1), j in 1:size(p_xy,2))
        return I
    end

    # mRMR feature selection algorithm
    function mrmr_rank(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer}; qbins::Int=5)
		println("    [mRMR] Starting...")
        Z = discretize_matrix(X; qbins=qbins)
        yb = y .+ 1
        nF = size(Z, 2)
        
        Ixy = [mi_discrete(view(Z,:,j), yb) for j in 1:nF]
        
        selected = Int[]; remaining = Set(1:nF)
        
        if isempty(Ixy) return selected, Ixy end
        first_feat = argmax(Ixy)
        push!(selected, first_feat); delete!(remaining, first_feat)

        while !isempty(remaining)
            best_j, best_score = -1, -Inf
            for j in remaining
                redundancy = mean(mi_discrete(view(Z,:,j), view(Z,:,s)) for s in selected)
                score = Ixy[j] - redundancy
                if score > best_score
                    best_score, best_j = score, j
                end
            end
            push!(selected, best_j); delete!(remaining, best_j)
        end
		println("    [mRMR] Finished.")
        return selected, Ixy
    end
	
	"✅ Helper functions defined successfully. You can now proceed to the 'CV Engine' cell."
end



# ╔═╡ 652bfb25-6e92-4f26-9a2e-6242449f00af
# --- Part 2: Nested Cross-Validation Engine ---
begin
    function inner_select_config(Xtr, ytr, subj_tr; n_inner=3, k_list=[20, 40, 80],
                                 svm_grid=[(1.0, 0.1), (10.0, 0.05)],
                                 gbdt_grid=[(0.05, 5, 300), (0.1, 3, 200)])
        println("  -- Starting Inner CV Loop --")
        inner_folds = create_subject_folds(subj_tr, n_inner; seed=123)
        best_score, best_mcc, best_config = -Inf, -Inf, nothing

        configs = vcat(
            [(:svm, k, p) for k in k_list for p in svm_grid],
            [(:gbdt, k, p) for k in k_list for p in gbdt_grid]
        )
        
        for (i, (model_sym, k, params)) in enumerate(configs)
            println("    [Inner CV] Config $i/$(length(configs)): $model_sym, k=$k, p=$params")
            fold_auprcs, fold_mccs = [], []
            for f in 1:n_inner
                val_mask = inner_folds .== f; train_mask = .!val_mask
                X_train_inner, y_train_inner = Xtr[train_mask, :], ytr[train_mask]
                X_val_inner, y_val_inner = Xtr[val_mask, :], ytr[val_mask]
                
                mrmr_features, _ = mrmr_rank(X_train_inner, y_train_inner)
                top_k = mrmr_features[1:min(k, length(mrmr_features))]
                
                model = model_sym == :svm ? SVM_RBF(C=params[1], γ=params[2]) : GBDT(η=params[1], depth=params[2], rounds=params[3])
                
                # --- FIX: Calculate n_pos and n_neg on separate lines ---
                n_pos = sum(y_train_inner .== 1)
                n_neg = length(y_train_inner) - n_pos
                weights = ones(length(y_train_inner)); 
                if n_pos > 0 && n_neg > 0 
                    weights[y_train_inner .== 1] .= n_neg / n_pos 
                end

                scores = train_predict_scores!(model, X_train_inner[:, top_k], y_train_inner, X_val_inner[:, top_k], weights=weights)
                
                _, _, auprc = pr_curve(y_val_inner, scores)
                _, mcc_val = best_threshold_mcc(y_val_inner, scores)
                push!(fold_auprcs, auprc); push!(fold_mccs, mcc_val)
				println("      > Inner Fold $f: AUPRC=$(round(auprc, digits=3)), MCC=$(round(mcc_val, digits=3))")
            end
            
            mean_auprc, mean_mcc = mean(fold_auprcs), mean(fold_mccs)
			println("    [Inner CV] Avg Metrics: AUPRC=$(round(mean_auprc, digits=3)), MCC=$(round(mean_mcc, digits=3))")

            if mean_auprc > best_score || (isapprox(mean_auprc, best_score) && mean_mcc > best_mcc)
                best_score, best_mcc = mean_auprc, mean_mcc
                best_config = (who=model_sym, k=k, params=params)
				println("    **** NEW BEST CONFIG FOUND ****")
                println("    [Inner CV] Best Score: AUPRC=$(round(best_score, digits=3)), MCC=$(round(best_mcc, digits=3))")
                println("    [Inner CV] Best Config: $(best_config)")
            end
        end
        println("  -- Inner CV Complete. Final Best Config: $(best_config) --")
        return best_config
    end

    function nested_cv_run(X, y, subject_id; n_outer=5, seed=42)
        outer_folds = create_subject_folds(subject_id, n_outer; seed=seed)
        results = []
        for f in 1:n_outer
            println("\n" * "="^50 * "\nOUTER FOLD $f/$n_outer\n" * "="^50)
            try
                test_mask = outer_folds .== f; train_mask = .!test_mask
                X_train, y_train, subj_train = X[train_mask,:], y[train_mask], subject_id[train_mask]
                X_test, y_test = X[test_mask,:], y[test_mask]
                
                best_config = inner_select_config(X_train, y_train, subj_train)
                if best_config === nothing; 
                    println("!!! WARNING: No best configuration found in inner loop for Outer Fold $f. Skipping fold.")
					continue; 
				end

				println("  [Outer Fold $f] Best config from inner loop: $(best_config)")
                println("  [Outer Fold $f] Retraining on full train set and evaluating on test set...")

                mrmr_features, _ = mrmr_rank(X_train, y_train)
                top_k = mrmr_features[1:min(best_config.k, end)]
                
                model = best_config.who == :svm ? SVM_RBF(C=best_config.params[1], γ=best_config.params[2]) : GBDT(η=best_config.params[1], depth=best_config.params[2], rounds=best_config.params[3])
                
                # --- FIX: Calculate n_pos and n_neg on separate lines ---
                n_pos = sum(y_train .== 1)
                n_neg = length(y_train) - n_pos
                weights = ones(length(y_train))
                if n_pos > 0 && n_neg > 0
                    weights[y_train .== 1] .= n_neg / n_pos 
                end

                scores = train_predict_scores!(model, X_train[:, top_k], y_train, X_test[:, top_k], weights=weights)
                
                τ_opt, _ = best_threshold_mcc(y_test, scores)
                prec, rec, auprc = pr_curve(y_test, scores)
                mcc_val = mcc(confusion_at(y_test, scores, τ_opt)...)
                
				println("  [Outer Fold $f] RESULT: AUPRC=$(round(auprc, digits=4)), MCC=$(round(mcc_val, digits=4)), Optimal τ=$(round(τ_opt, digits=3))")

                result_tuple = (model=best_config.who, k=best_config.k, AUPRC=auprc, MCC=mcc_val, threshold=τ_opt, features=top_k, pr_data=(prec,rec), mcc_curve=([t for t in 0:0.01:1], [mcc(confusion_at(y_test, scores, t)...) for t in 0:0.01:1]))
                push!(results, result_tuple)
				println("  [Outer Fold $f] Stored results: $(result_tuple)")

            catch e
                println("!!! ERROR in Outer Fold $f: $e"); showerror(stdout, e, catch_backtrace());
            end
        end
		println("\n" * "="^50 * "\nNESTED CV FINISHED\n" * "="^50)
        return results
    end
	
	"✅ CV Engine functions defined successfully. You can now run the 'Task 3 - Main Execution' cell."
end



# ╔═╡ 2a6c54b3-425c-43a5-8b72-53f78d31b13d
# ╠═╡ disabled = true
#=╠═╡
# --- Part 3: Scaled-Down Main Execution for Fast Debugging (Self-Contained) ---
begin

    println("Starting SCALED-DOWN Task 3: Nested Cross-Validation...")
    println("This uses a reduced hyperparameter grid for quick testing.")

    # --- CV Engine Functions (included directly to ensure correctness) ---

    # ----- Inner model and feature selection -----
    function inner_select_config(
        Xtr::Matrix{Float64}, ytr::Vector{Int}, subj_tr;
        n_inner::Int = 3,
        k_list::Vector{Int}, svm_grid::Vector, gbdt_grid::Vector
    )
        inner = create_subject_folds(subj_tr, n_inner; seed=777)
        rank_on(mask) = mrmr_rank(Xtr[mask, :], ytr[mask]; qbins=5)[1]
        best = (score=-Inf, mcc=-Inf, who=:none, k=0, params=nothing)

        configs = vcat(
            [(:svm,  k, p) for k in k_list for p in svm_grid],
            [(:gbdt, k, p) for k in k_list for p in gbdt_grid]
        )
        println("    -- Starting Inner CV Loop with $(length(configs)) configurations --")

        for (i, (sym, k, params)) in enumerate(configs)
            println("      [Inner CV] Config $i/$(length(configs)): $sym, k=$k, p=$params")
            aus = Float64[]
            mcs = Float64[]

            for f in 1:n_inner
                va = inner .== f
                tr = .!va
                if sum(va) == 0
                    continue
                end

                idxs   = rank_on(tr)[1:min(k, size(Xtr, 2))]
                Xtr_f  = Xtr[tr, idxs]
                Xva_f  = Xtr[va, idxs]
                ytr_f  = ytr[tr]
                yva_f  = ytr[va]

                model = sym == :svm  ? SVM_RBF(C=params[1], γ=params[2]) :
                        GBDT(η=params[1], depth=params[2], rounds=params[3])

                scores = train_predict_scores!(model, Xtr_f, ytr_f, Xva_f)
                _, _, au = pr_curve(yva_f, scores)
                _, M     = best_threshold_mcc(yva_f, scores)

                push!(aus, au)
                push!(mcs, M)
            end

            if isempty(aus)
                continue
            end

            μau = mean(aus)
            μm  = mean(mcs)

            if μau > best.score || (isapprox(μau, best.score; atol=1e-9) && μm > best.mcc)
                best = (score=μau, mcc=μm, who=sym, k=k, params=params)
                println("    **** NEW BEST CONFIG FOUND ****")
                println("    [Inner CV] Best: AUPRC=$(round(best.score, digits=3)), MCC=$(round(best.mcc, digits=3)) -> :$(best.who), k=$(best.k), p=$(best.params)")
            end
        end

        return best
    end

    # ----- Nested outer CV -----
    function nested_cv_run(
        X::Matrix{Float64}, y::Vector{Int}, subject_id;
        n_outer::Int=5, seed::Int=42, k_list, svm_grid, gbdt_grid
    )
        folds = create_subject_folds(subject_id, n_outer; seed=seed)

        outer_stats = DataFrame(
            Fold=Int[], Model=String[], k=Int[],
            AUPRC=Float64[], MCC=Float64[], tau=Float64[]
        )

        sel_sets = Dict{String, Vector{Vector{Int}}}(
            "svm"  => Vector{Vector{Int}}(),
            "gbdt" => Vector{Vector{Int}}()
        )
        pr_store = Dict{String, Vector{Tuple{Vector{Float64}, Vector{Float64}}}}(
            "svm"  => Vector{Tuple{Vector{Float64}, Vector{Float64}}}(),
            "gbdt" => Vector{Tuple{Vector{Float64}, Vector{Float64}}}(),
        )
        thr_store = Dict{String, Vector{Tuple{Vector{Float64}, Vector{Float64}}}}(
            "svm"  => Vector{Tuple{Vector{Float64}, Vector{Float64}}}(),
            "gbdt" => Vector{Tuple{Vector{Float64}, Vector{Float64}}}(),
        )

        for f in 1:n_outer
            println("\n" * "="^50 * "\nOUTER FOLD $f/$n_outer\n" * "="^50)
            te = folds .== f
            tr = .!te

            Xtr = X[tr, :]
            Xte = X[te, :]
            ytr = y[tr]
            yte = y[te]
            subj_tr = subject_id[tr]

            try
                best = inner_select_config(Matrix(Xtr), ytr, subj_tr; k_list=k_list, svm_grid=svm_grid, gbdt_grid=gbdt_grid)

                if best.who == :none || best.k <= 0
                    println("    No valid configuration found for fold $f, skipping")
                    continue
                end

                rank_full = mrmr_rank(Matrix(Xtr), ytr)[1]
                idxs_full = rank_full[1:best.k]
                model_name_str = String(best.who)
                push!(sel_sets[model_name_str], idxs_full)

                model = best.who == :svm ?
                        SVM_RBF(C=best.params[1], γ=best.params[2]) :
                        GBDT(η=best.params[1], depth=best.params[2], rounds=best.params[3])

                scores_te = train_predict_scores!(
                    model,
                    Matrix(Xtr[:, idxs_full]), ytr,
                    Matrix(Xte[:, idxs_full])
                )

                prec, rec, au = pr_curve(yte, scores_te)
                push!(pr_store[model_name_str], (rec, prec))

                grid   = 0.0:0.01:1.0
                mvals  = [mcc(confusion_at(yte, scores_te, t)...) for t in grid]
                push!(thr_store[model_name_str], (collect(grid), mvals))

                t_idx  = argmax(mvals)
                tau    = collect(grid)[t_idx]
                best_m = mvals[t_idx]

                push!(outer_stats, (Fold=f, Model=model_name_str, k=best.k, AUPRC=au, MCC=best_m, tau=tau))
            catch e
                println("!!! ERROR in Outer Fold $f: $e")
                showerror(stdout, e, catch_backtrace())
                println()
            end
        end

        println("\n" * "="^50 * "\nNESTED CV FINISHED\n" * "="^50)
        return outer_stats, sel_sets, pr_store, thr_store
    end

    # --- Execution Logic ---

    if !@isdefined(X_all)
        X_all = hcat(values(X_schemes)...)
        println("Combined feature matrix created with dimensions: $(size(X_all))")
    end

    k_list_scaled     = [20, 50]
    svm_grid_scaled   = [(C, γ) for C in [1.0, 10.0], γ in [0.05, 0.1]]
    gbdt_grid_scaled  = [(η, depth, rounds) for η in [0.05], depth in [5], rounds in [200]]

    scaled_results, scaled_sets, scaled_pr, scaled_thr = try
        nested_cv_run(
            X_all, y, subject_id;
            n_outer=3, seed=42,
            k_list=k_list_scaled,
            svm_grid=vec(svm_grid_scaled),
            gbdt_grid=vec(gbdt_grid_scaled)
        )
    catch e
        println("\n" * "!"^80 * "\n!!! EXECUTION FAILED !!!\n" * "-"^80)
        println("Fault Logged:\nError Type: $(typeof(e))\nMessage: $(sprint(showerror, e))")
        println("!"^80 * "\n")
        println("\n--- DETAILED STACK TRACE ---")
        showerror(stdout, e, catch_backtrace())
        println("\n" * "-"^28)
        (nothing, nothing, nothing, nothing)
    end

    if scaled_results !== nothing
        if isempty(scaled_results)
            println("\n" * "⚠️"^40 * "\n⚠️ EXECUTION COMPLETED WITH NO RESULTS ⚠️\n" * "-"^80)
            println("The cross-validation process finished without successfully training any models.")
            println("Please check the logs for errors within the outer fold loops.")
            println("⚠️"^40 * "\n")
        else
            println("\n--- SCALED-DOWN Task 3 Execution Finished ---")
            display(summarize_stats(scaled_results))
            println("\nFeature Stability from scaled-down run:")
            print_stability(scaled_sets)
            println("\n✅ Scaled-down run complete. You can now run the visualization cell.")
        end
    else
        println("\n❌ Scaled-down run failed. Please see the error report above.")
    end
end

  ╠═╡ =#

# ╔═╡ 1ca887db-7027-4ae9-bc57-6f3b4cee3c16
# ╠═╡ disabled = true
#=╠═╡

begin
    have_data = @isdefined(scaled_results) && scaled_results !== nothing && !isempty(scaled_results) &&
                @isdefined(scaled_pr) && @isdefined(scaled_thr)

    if have_data
        # --- PR curves (scaled) ---
        pr_plot = let
            p = plot(layout=(1,2), xlabel="Recall", ylabel="Precision", legend=:bottomleft, titlefontsize=10)
            for (j, name) in enumerate(["svm", "gbdt"])
                curves = get(scaled_pr, name, [])
                if isempty(curves)
                    annotate!(p[j], 0.5, 0.5, text("No data for $(uppercase(name))", 10, :gray))
                else
                    # scaled_pr stores tuples as (rec, prec)
                    for (i, (rec, prec)) in enumerate(curves)
                        plot!(p[j], rec, prec, label="Fold $i", lw=1.5)
                    end
                end
                title!(p[j], "PR Curves - $(uppercase(name)) (scaled)")
                xlims!(p[j], 0, 1)
                ylims!(p[j], 0, 1)
            end
            p
        end

        # --- Threshold vs MCC (scaled) ---
        mcc_plot = let
            p = plot(layout=(1,2), xlabel="Threshold", ylabel="MCC", legend=:bottomright, titlefontsize=10)
            for (j, name) in enumerate(["svm", "gbdt"])
                curves = get(scaled_thr, name, [])
                fold_stats = filter(row -> lowercase(row.Model) == name, scaled_results)
                if isempty(curves)
                    annotate!(p[j], 0.5, 0.5, text("No data for $(uppercase(name))", 10, :gray))
                else
                    for (i, (grid, mvals)) in enumerate(curves)
                        plot!(p[j], grid, mvals, label="Fold $(i)", lw=1.5)
                        if !isempty(fold_stats) && i <= nrow(fold_stats)
                            vline!(p[j], [fold_stats.tau[i]], ls=:dash, label="τ* F$(fold_stats.Fold[i])")
                        end
                    end
                end
                title!(p[j], "Threshold vs MCC - $(uppercase(name)) (scaled)")
                xlims!(p[j], 0, 1)
            end
            p
        end

        plot(pr_plot, mcc_plot, layout=(2,1), size=(1100,800))
    else
        md"""
        Run the scaled Task 3 execution cell to create `scaled_results`, `scaled_pr`, and `scaled_thr`.
        """
    end
end

  ╠═╡ =#

# ╔═╡ 46bf3560-b7c2-4a04-bb6c-c878c199edb8
md"""
## Task 4
"""

# ╔═╡ 23959308-43b0-49a7-a89d-027c6377f792
# t4_engine_glue
begin
    # Reuse existing Task 3 helpers, only add g_* names if missing

    if !@isdefined(g_mrmr_rank)
        g_mrmr_rank(X, y; qbins::Int=5) = mrmr_rank(X, y; qbins=qbins)
    end
    if !@isdefined(g_pr_curve)
        g_pr_curve(y, scores) = pr_curve(y, scores)   # returns (prec, rec, au)
    end
    if !@isdefined(g_best_threshold_mcc)
        g_best_threshold_mcc(y, scores; grid=0.0:0.005:1.0) = best_threshold_mcc(y, scores; grid=grid)
    end
    if !@isdefined(g_mcc)
        g_mcc(tp, tn, fp, fn) = mcc(tp, tn, fp, fn)
    end
    if !@isdefined(g_confusion_at)
        g_confusion_at(y, scores, τ) = confusion_at(y, scores, τ)
    end
    if !@isdefined(g_train_predict_scores!)
        g_train_predict_scores!(model, Xtr, ytr, Xva) = train_predict_scores!(model, Xtr, ytr, Xva)
    end
    if !@isdefined(g_SVM_RBF)
        g_SVM_RBF(; C::Float64=1.0, γ::Float64=0.1) = SVM_RBF(C=C, γ=γ)
    end
    if !@isdefined(g_GBDT)
        g_GBDT(; η=0.05, depth=5, rounds=300) = GBDT(η=η, depth=depth, rounds=rounds)
    end
    if !@isdefined(g_create_subject_folds)
        g_create_subject_folds(subj, n; seed=777) = create_subject_folds(subj, n; seed=seed)
    end

    if !@isdefined(g_inner_select_config)
        function g_inner_select_config(
            Xtr::AbstractMatrix{<:Real}, ytr::Vector{<:Integer}, subj_tr;
            n_inner::Int=3,
            k_list::Vector=[20, 50],
            svm_grid::Vector=[(1.0, 0.1)],
            gbdt_grid::Vector=[(0.05, 5, 200)],
        )
            inner = g_create_subject_folds(subj_tr, n_inner; seed=777)
            rank_on(mask) = g_mrmr_rank(Matrix(Xtr[mask, :]), ytr[mask]; qbins=5)[1]
            best = (score=-Inf, mcc=-Inf, who=:none, k=0, params=nothing)

            configs = vcat(
                [(:svm,  k, p) for k in k_list for p in svm_grid],
                [(:gbdt, k, p) for k in k_list for p in gbdt_grid],
            )

            for (sym, k, params) in configs
                aus = Float64[]
                mcs = Float64[]
                for f in 1:n_inner
                    va = inner .== f
                    tr = .!va
                    if sum(va) == 0
                        continue
                    end
                    idxs  = rank_on(tr)[1:min(k, size(Xtr, 2))]
                    Xtr_f = Matrix(Xtr[tr, idxs])
                    Xva_f = Matrix(Xtr[va, idxs])
                    ytr_f = ytr[tr]
                    yva_f = ytr[va]

                    model = sym == :svm ?
                        g_SVM_RBF(C=params[1], γ=params[2]) :
                        g_GBDT(η=params[1], depth=params[2], rounds=params[3])

                    scores = g_train_predict_scores!(model, Xtr_f, ytr_f, Xva_f)
                    _, _, au = g_pr_curve(yva_f, scores)
                    _, M     = g_best_threshold_mcc(yva_f, scores)
                    push!(aus, au); push!(mcs, M)
                end

                if !isempty(aus)
                    μau = mean(aus)
                    μm  = mean(mcs)
                    if μau > best.score || (isapprox(μau, best.score; atol=1e-9) && μm > best.mcc)
                        best = (score=μau, mcc=μm, who=sym, k=k, params=params)
                    end
                end
            end
            return best
        end
    end

    if !@isdefined(g_nested_cv_run)
        function g_nested_cv_run(
            X::AbstractMatrix{<:Real}, y::Vector{<:Integer}, subject_id;
            n_outer::Int=3, seed::Int=42,
            k_list=[20, 50],
            svm_grid=[(1.0, 0.05), (1.0, 0.1), (10.0, 0.05), (10.0, 0.1)],
            gbdt_grid=[(0.05, 5, 200)],
        )
            folds = g_create_subject_folds(subject_id, n_outer; seed=seed)

            outer_stats = DataFrame(Fold=Int[], Model=String[], k=Int[],
                                    AUPRC=Float64[], MCC=Float64[], tau=Float64[])
            sel_sets = Dict("svm"=>Vector{Vector{Int}}(), "gbdt"=>Vector{Vector{Int}}())
            pr_store = Dict("svm"=>Vector{Tuple{Vector{Float64}, Vector{Float64}}}(),
                            "gbdt"=>Vector{Tuple{Vector{Float64}, Vector{Float64}}}())
            thr_store = Dict("svm"=>Vector{Tuple{Vector{Float64}, Vector{Float64}}}(),
                             "gbdt"=>Vector{Tuple{Vector{Float64}, Vector{Float64}}}())

            for f in 1:n_outer
                te = folds .== f
                tr = .!te

                Xtr = Matrix(X[tr, :])
                Xte = Matrix(X[te, :])
                ytr = y[tr]
                yte = y[te]
                subj_tr = subject_id[tr]

                best = g_inner_select_config(
                    Xtr, ytr, subj_tr;
                    n_inner=3, k_list=k_list, svm_grid=svm_grid, gbdt_grid=gbdt_grid
                )
                if best.who == :none || best.k <= 0
                    continue
                end

                rank_full = g_mrmr_rank(Xtr, ytr; qbins=5)[1]
                idxs_full = rank_full[1:best.k]
                push!(sel_sets[String(best.who)], idxs_full)

                model = best.who == :svm ?
                    g_SVM_RBF(C=best.params[1], γ=best.params[2]) :
                    g_GBDT(η=best.params[1], depth=best.params[2], rounds=best.params[3])

                scores_te = g_train_predict_scores!(
                    model,
                    Matrix(Xtr[:, idxs_full]), ytr,
                    Matrix(Xte[:, idxs_full])
                )

                prec, rec, au = g_pr_curve(yte, scores_te)   # store as (prec, rec)
                push!(pr_store[String(best.who)], (prec, rec))

                grid  = 0.0:0.01:1.0
                mvals = [g_mcc(g_confusion_at(yte, scores_te, t)...) for t in grid]
                push!(thr_store[String(best.who)], (collect(grid), mvals))

                t_idx  = argmax(mvals)
                tau    = collect(grid)[t_idx]
                best_m = mvals[t_idx]

                push!(outer_stats, (Fold=f, Model=uppercase(String(best.who)), k=best.k,
                                    AUPRC=au, MCC=best_m, tau=tau))
            end

            return outer_stats, sel_sets, pr_store, thr_store
        end
    end
end


# ╔═╡ 3752bcd2-7e98-4a3d-b2f5-0584417513c8
# t4_shim_define
begin
    function g_run_gender_specific_scaled(
        X_schemes, gender, y, subject_id;
        n_outer=3,
        seed_male=123, seed_female=456,
        k_list=[20, 50],
        svm_grid=[(1.0, 0.05), (1.0, 0.1), (10.0, 0.05), (10.0, 0.1)],
        gbdt_grid=[(0.05, 5, 200)],
    )
        # pick a CV engine that exists
        _cv = @isdefined(g_nested_cv_run) ? g_nested_cv_run :
              (@isdefined(nested_cv_run) ? nested_cv_run : nothing)
        _cv === nothing && error("No nested CV engine found. Define `g_nested_cv_run` (Task 4 helpers) or `nested_cv_run` (Task 3).")

        X_all = hcat(values(X_schemes)...)

        # males
        male_mask = gender .== 1
        X_m = Matrix{Float64}(X_all[male_mask, :])
        y_m = Int.(y[male_mask])
        sid_m = subject_id[male_mask]
        male_stats, male_sets, male_pr, male_thr =
            _cv(X_m, y_m, sid_m;
                n_outer=n_outer, seed=seed_male,
                k_list=k_list, svm_grid=vec(svm_grid), gbdt_grid=vec(gbdt_grid))

        # females
        female_mask = gender .== 0
        X_f = Matrix{Float64}(X_all[female_mask, :])
        y_f = Int.(y[female_mask])
        sid_f = subject_id[female_mask]
        female_stats, female_sets, female_pr, female_thr =
            _cv(X_f, y_f, sid_f;
                n_outer=n_outer, seed=seed_female,
                k_list=k_list, svm_grid=vec(svm_grid), gbdt_grid=vec(gbdt_grid))

        return (
            male   = (results=male_stats,   sets=male_sets,   pr=male_pr,   thr=male_thr),
            female = (results=female_stats, sets=female_sets, pr=female_pr, thr=female_thr),
        )
    end
end


# ╔═╡ a805b547-a9cb-44c7-8a43-aecada6693bb
# t4_fix_runner
begin
    """
    Run Task 4 on male and female subsets with a small grid.
    Returns named tuples for each subset: (results, sets, pr, thr)
    """
    function g_run_gender_specific_scaled(
        X_schemes::AbstractDict,
        gender::AbstractVector{<:Integer},
        y::AbstractVector{<:Integer},
        subject_id::AbstractVector;
        n_outer::Int = 3,
        seed_male::Int = 123,
        seed_female::Int = 456,
        k_list::Vector{Int} = [20, 50],
        svm_grid::Vector{Tuple{Float64,Float64}} = [(1.0,0.05),(1.0,0.1),(10.0,0.05),(10.0,0.1)],
        gbdt_grid::Vector{Tuple{Float64,Int,Int}} = [(0.05,5,200)]
    )
        X_all = hcat(values(X_schemes)...)

        male_mask   = gender .== 1
        female_mask = gender .== 0

        X_male  = Matrix{Float64}(X_all[male_mask, :])
        y_male  = Int.(y[male_mask])
        s_male  = subject_id[male_mask]

        X_fem   = Matrix{Float64}(X_all[female_mask, :])
        y_fem   = Int.(y[female_mask])
        s_fem   = subject_id[female_mask]

        male_stats, male_sets, male_pr, male_thr = g_nested_cv_run(
            X_male, y_male, s_male;
            n_outer=n_outer, seed=seed_male,
            k_list=k_list, svm_grid=svm_grid, gbdt_grid=gbdt_grid
        )

        female_stats, female_sets, female_pr, female_thr = g_nested_cv_run(
            X_fem, y_fem, s_fem;
            n_outer=n_outer, seed=seed_female,
            k_list=k_list, svm_grid=svm_grid, gbdt_grid=gbdt_grid
        )

        return (
            male   =(results=male_stats,   sets=male_sets,   pr=male_pr,   thr=male_thr),
            female =(results=female_stats, sets=female_sets, pr=female_pr, thr=female_thr),
        )
    end
end


# ╔═╡ ed9c8f93-c520-4712-af5b-c372fef4c6a8
# t4_run_scaled
begin
    # Case-robust gender column grab and normalization to 0/1
    gender_raw =
        hasproperty(df, :Gender) ? df.Gender :
        hasproperty(df, :gender) ? df.gender :
        error("Gender column not found. Tried :Gender and :gender")

    gender_vec =
        eltype(gender_raw) <: Integer ? Int.(gender_raw) :
        eltype(gender_raw) <: AbstractFloat ? round.(Int, gender_raw) :
        [let s = lowercase(string(x))
             s in ("m","male","1","true") ? 1 :
             s in ("f","female","0","false") ? 0 :
             error("Unparsed gender value: $(x)")
         end for x in gender_raw]

    g_out = g_run_gender_specific_scaled(X_schemes, gender_vec, y, subject_id)

    g_male_results   = g_out.male.results
    g_male_sets      = g_out.male.sets
    g_male_pr        = g_out.male.pr      # stored as (prec, rec)
    g_male_thr       = g_out.male.thr

    g_female_results = g_out.female.results
    g_female_sets    = g_out.female.sets
    g_female_pr      = g_out.female.pr    # stored as (prec, rec)
    g_female_thr     = g_out.female.thr

    display(g_male_results)
    display(g_female_results)
end


# ╔═╡ e299f27a-576f-44b3-b49b-9ad6728cb446
# t4_visualisations
begin
    # PR curves for males and females. g_* stores (prec, rec), so plot rec on x, prec on y.
    function g_plot_pr_curves_gender(pr_store_m, pr_store_f; title_suffix="")
        p = plot(layout=(1,2), xlabel="Recall", ylabel="Precision", legend=:bottomleft)
        for (j, name, store) in ((1, "Male",   pr_store_m),
                                 (2, "Female", pr_store_f))
            curves_svm  = get(store, "svm",  [])
            curves_gbdt = get(store, "gbdt", [])
            if isempty(curves_svm) && isempty(curves_gbdt)
                annotate!(p[j], 0.5, 0.5, text("No PR data", 10, :grey))
            else
                for (i, (prec, rec)) in enumerate(curves_svm)
                    plot!(p[j], rec, prec, label="SVM fold $i", lw=1.6)
                end
                for (i, (prec, rec)) in enumerate(curves_gbdt)
                    plot!(p[j], rec, prec, label="GBDT fold $i", lw=1.6, ls=:dash)
                end
            end
            title!(p[j], "PR Curves $title_suffix — $name")
            xlims!(p[j], (0,1)); ylims!(p[j], (0,1))
        end
        p
    end

    # Threshold vs MCC for males and females
    function g_plot_thr_mcc_gender(thr_store_m, stats_m, thr_store_f, stats_f; title_suffix="")
        p = plot(layout=(1,2), xlabel="Threshold", ylabel="MCC", legend=:bottomright)

        for (j, name, thr_store, stats_df) in ((1, "Male",   thr_store_m, stats_m),
                                               (2, "Female", thr_store_f, stats_f))
            for model_key in ("svm", "gbdt")
                curves = get(thr_store, model_key, [])
                fold_stats = filter(row -> lowercase(row.Model) == model_key, stats_df)
                for (i, (grid, mvals)) in enumerate(curves)
                    plot!(p[j], grid, mvals,
                          label="$(uppercase(model_key)) fold $(i)", lw=1.6)
                    if i <= nrow(fold_stats)
                        vline!(p[j], [fold_stats.tau[i]], ls=:dash, label="τ* F$(fold_stats.Fold[i])")
                    end
                end
            end
            title!(p[j], "Threshold vs MCC $title_suffix — $name")
            xlims!(p[j], (0,1))
        end
        p
    end

    have_gender =
        @isdefined(g_male_results) && @isdefined(g_female_results) &&
        !isempty(g_male_results) && !isempty(g_female_results) &&
        @isdefined(g_male_pr) && @isdefined(g_female_pr) &&
        @isdefined(g_male_thr) && @isdefined(g_female_thr)

    if have_gender
        prp  = g_plot_pr_curves_gender(g_male_pr, g_female_pr; title_suffix="(Task 4 scaled)")
        mccp = g_plot_thr_mcc_gender(g_male_thr, g_male_results, g_female_thr, g_female_results; title_suffix="(Task 4 scaled)")
        plot(prp, mccp, layout=(2,1), size=(1100,900))
    else
        md"Run the Task 4 runner cell first."
    end
end


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
EvoTrees = "f6006082-12f8-11e9-0c9c-0d5d367ab1e5"
LIBSVM = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJBase = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
MLJDecisionTreeInterface = "c6f25543-311c-4c74-83dc-3ea6d1015661"
MLJLIBSVMInterface = "61c7150f-6c77-4bb1-949c-13197eac2a52"
ManifoldLearning = "06eb3307-b2af-5a2a-abea-d33192699d32"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
NearestNeighborModels = "636a865e-7cf4-491e-846c-de09b730eb36"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
TSne = "24678dba-d5e9-5843-a4c6-250288b04835"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.0"
EvoTrees = "~0.17.4"
LIBSVM = "~0.8.1"
MLJ = "~0.20.0"
MLJBase = "~1.7.0"
MLJDecisionTreeInterface = "~0.4.2"
MLJLIBSVMInterface = "~0.2.1"
ManifoldLearning = "~0.9.0"
MultivariateStats = "~0.9.1"
NearestNeighborModels = "~0.2.3"
Plots = "~1.40.16"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.8"
TSne = "~1.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "1dbdce73202a4882d6800a5397ef1f0b7e2df5b2"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "678eb18590a8bc6674363da4d5faa4ac09c40a18"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.5.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b5bb4dc6248fde467be2a863eb8452993e74d402"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BSON]]
git-tree-sha1 = "4c3e506685c527ac6a54ccc0c8c76fd6f91b42fb"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.9"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes"]
git-tree-sha1 = "926862f549a82d6c3a7145bc7f1adff2a91a39f0"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.15"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "06ee8d1aa558d2833aa799f6f0b31b30cada405f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.2"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "8010b6bb3388abe68d95743dcbea77650bb2eddf"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.3"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "3a3dfb30697e96a440e4149c8c51bf32f818c0f3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.17.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DecisionTree]]
deps = ["AbstractTrees", "DelimitedFiles", "LinearAlgebra", "Random", "ScikitLearnBase", "Statistics"]
git-tree-sha1 = "526ca14aaaf2d5a0e242f3a8a7966eb9065d7d78"
uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
version = "0.12.4"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c2614fa3aafe03d1a44b8e16508d9be718b8095a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.89"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.EvoTrees]]
deps = ["BSON", "CategoricalArrays", "Distributions", "MLJModelInterface", "NetworkLayout", "Random", "RecipesBase", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "a5e04a7b056002cba9583012d24a66a6c0ae7735"
uuid = "f6006082-12f8-11e9-0c9c-0d5d367ab1e5"
version = "0.17.4"

    [deps.EvoTrees.extensions]
    EvoTreesCUDAExt = "CUDA"

    [deps.EvoTrees.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "1828eb7275491981fa5f1752a5e126e8f26f8741"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.17"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "27299071cc29e409488ada41ec7643e0ab19091f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.17+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

    [deps.GeometryBasics.weakdeps]
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "35fbd0cefb04a516104b8e183ce0df11b70a3f1a"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "c5abfa0ae0aaee162a3fbb053c13ecda39be545b"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8594fac023c5ce1ef78260f24d1ad18b4327b420"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.4"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "e663925ebc3d93c1150a7570d114f9ea2f664726"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "4efa9cec6f308e0f492ea635421638bff81cf6f8"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.36"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LIBLINEAR]]
deps = ["Libdl", "SparseArrays", "liblinear_jll"]
git-tree-sha1 = "2cd424d3bf9b36098009df5b1f399614c12b2ee4"
uuid = "2d691ee1-e668-5016-a719-b2531b85e0f5"
version = "0.7.1"

[[deps.LIBSVM]]
deps = ["LIBLINEAR", "LinearAlgebra", "ScikitLearnBase", "SparseArrays", "libsvm_jll"]
git-tree-sha1 = "9016c6032aac779b13bbd1b3ce997606a6eb7a2b"
uuid = "b1bec4e5-fd48-53fe-b0cb-9723c09d164b"
version = "0.8.1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "4f34eaabe49ecb3fb0d58d6015e32fd31a733199"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.8"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "825289d43c753c7f1bf9bed334c253e9913997f8"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.9.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LearnAPI]]
deps = ["InteractiveUtils", "Statistics"]
git-tree-sha1 = "ec695822c1faaaa64cee32d0b21505e1977b4809"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "0.1.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLFlowClient]]
deps = ["Dates", "FilePathsBase", "HTTP", "JSON", "ShowCases", "URIs", "UUIDs"]
git-tree-sha1 = "5cc2a5453856e79f4772269fbe6b19fcdcba391a"
uuid = "64a0f543-368b-4a9a-827a-e71edb2a0b83"
version = "0.4.7"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "LinearAlgebra", "MLJBase", "MLJEnsembles", "MLJFlow", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "Reexport", "ScientificTypes", "StatisticalMeasures", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "58d17a367ee211ade6e53f83a9cc5adf9d26f833"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.20.0"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LearnAPI", "LinearAlgebra", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "RecipesBase", "Reexport", "ScientificTypes", "Serialization", "StatisticalMeasuresBase", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "6f45e12073bc2f2e73ed0473391db38c31e879c9"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "1.7.0"
weakdeps = ["StatisticalMeasures"]

    [deps.MLJBase.extensions]
    DefaultMeasuresExt = "StatisticalMeasures"

[[deps.MLJDecisionTreeInterface]]
deps = ["CategoricalArrays", "DecisionTree", "MLJModelInterface", "Random", "Tables"]
git-tree-sha1 = "90ef4d3b6cacec631c57cc034e1e61b4aa0ce511"
uuid = "c6f25543-311c-4c74-83dc-3ea6d1015661"
version = "0.4.2"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatisticalMeasuresBase", "StatsBase"]
git-tree-sha1 = "84a5be55a364bb6b6dc7780bbd64317ebdd3ad1e"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.4.3"

[[deps.MLJFlow]]
deps = ["MLFlowClient", "MLJBase", "MLJModelInterface"]
git-tree-sha1 = "dc0de70a794c6d4c1aa4bde8196770c6b6e6b550"
uuid = "7b7b8358-b45c-48ea-a8ef-7ca328ad328f"
version = "0.2.0"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "ad16cfd261e28204847f509d1221a581286448ae"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.6.3"

[[deps.MLJLIBSVMInterface]]
deps = ["CategoricalArrays", "LIBSVM", "MLJModelInterface", "Statistics"]
git-tree-sha1 = "4a056d2384a906ac2a2d13ed25f9092107f53eb0"
uuid = "61c7150f-6c77-4bb1-949c-13197eac2a52"
version = "0.2.1"

[[deps.MLJModelInterface]]
deps = ["REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "66626f80d5807921045d539b4f7153b1d47c5f8a"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.1"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "410da88e0e6ece5467293d2c76b51b7c6df7d072"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.16.17"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase", "StatisticalMeasuresBase"]
git-tree-sha1 = "38aab60b1274ce7d6da784808e3be69e585dbbf6"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.8.8"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManifoldLearning]]
deps = ["Combinatorics", "Graphs", "LinearAlgebra", "MultivariateStats", "Random", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "4c5564c707899c3b6bc6d324b05e43eb7f277f2b"
uuid = "06eb3307-b2af-5a2a-abea-d33192699d32"
version = "0.9.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "4abc63cdd8dd9dd925d8e879cda280bedc8013ca"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.30"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighborModels]]
deps = ["Distances", "FillArrays", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "NearestNeighbors", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "e411143a8362926e4284a54e745972e939fbab78"
uuid = "636a865e-7cf4-491e-846c-de09b730eb36"
version = "0.2.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ca7e18198a166a1f3eb92a3650d53d94ed8ca8a1"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.22"

[[deps.NetworkLayout]]
deps = ["GeometryBasics", "LinearAlgebra", "Random", "Requires", "StaticArrays"]
git-tree-sha1 = "f7466c23a7c5029dc99e8358e7ce5d81a117c364"
uuid = "46757867-2c16-5918-afeb-47bfcb05e46a"
version = "0.4.10"
weakdeps = ["Graphs"]

    [deps.NetworkLayout.extensions]
    NetworkLayoutGraphsExt = "Graphs"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg", "Scratch"]
git-tree-sha1 = "63603b2b367107e87dbceda4e33c67aed17e50e0"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "87510f7292a2b21aeff97912b0898f9553cc5c2c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "55818b50883d7141bd98cdf5fc2f4ced96ee075f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.16"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "142ee93724a9c5d04d78df7006670a93ed1b244e"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "13c5103482a8ed1536a54c08d0e742ae3dca2d42"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.4"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "e1d5e16d0f65762396f9ca4644a5f4ddab8d452b"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "4d083ffec53dbd5097a6723b0699b175be2b61fb"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.1.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StatisticalMeasures]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Distributions", "LearnAPI", "LinearAlgebra", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "StatisticalMeasuresBase", "Statistics", "StatsBase"]
git-tree-sha1 = "c1d4318fa41056b839dfbb3ee841f011fa6e8518"
uuid = "a19d573c-0a75-4610-95b3-7071388c7541"
version = "0.1.7"

    [deps.StatisticalMeasures.extensions]
    LossFunctionsExt = "LossFunctions"
    ScientificTypesExt = "ScientificTypes"

    [deps.StatisticalMeasures.weakdeps]
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "e4f508cf3b3253f3eb357274fe36fb3332ca9896"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "88cf3587711d9ad0a55722d339a013c4c56c5bbc"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TSne]]
deps = ["Distances", "LinearAlgebra", "Printf", "ProgressMeter", "Statistics"]
git-tree-sha1 = "6f1dfbf9dad6958439816fa9c5fa20898203fdf4"
uuid = "24678dba-d5e9-5843-a4c6-250288b04835"
version = "1.3.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d2282232f8a4d71f79e85dc4dd45e5b12a6297fb"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.23.1"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "XML2_jll"]
git-tree-sha1 = "9300077d83c9eccfcac6a18b442d857d7e4f1684"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.23.1+2"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "c5bf2dad6a03dfef57ea0a170a1fe493601603f2"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.5+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.liblinear_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7f5f1953394b74739eaebd345f4515515a022a5b"
uuid = "275f1f90-abd2-5ca1-9ad8-abd4e3d66eb7"
version = "2.47.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libsvm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "Pkg"]
git-tree-sha1 = "7625dde5e9eab416c1cb791627f065ce55297eff"
uuid = "08558c22-525a-5d2a-acf6-0ac6658ffce4"
version = "3.25.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╠═1ca7d017-da7f-4d9d-9dd3-ebe327bdf140
# ╟─23550ef3-a47e-4d8d-952f-e63bc56b6cf8
# ╠═a32f2f31-42df-4410-86df-3ad3784fcf7e
# ╠═08de05af-d020-46c3-96ee-8fcdc8557466
# ╠═e8691046-2118-4f57-bab3-17bed354e713
# ╠═01efd005-143a-4618-b7fd-68c3652389e4
# ╠═95224368-0653-4c50-a31c-22bfc2d3eee7
# ╟─0e1503e1-89e6-4638-be96-268c540d6b8c
# ╠═e1b90593-00f0-4984-b0c0-f2ea78a8fedb
# ╠═af3237ee-28ea-43e4-b9fe-67cf4ad97d27
# ╠═99256a50-18d6-4e5c-a742-866a55ba0060
# ╟─b40fcb46-1101-4ecf-aaec-ff90686e12ca
# ╠═a657d8ca-fde8-4b87-a4da-3b3ae96c04ed
# ╠═5f72ca76-2e44-49f3-83b9-6cae21a8e690
# ╠═f8dcdd9a-9318-469c-b182-7a87c5cf816e
# ╠═8771a550-2183-4918-b191-11b97fb3e6b2
# ╠═afaf5667-9337-4837-aa33-f3886a7102ea
# ╠═19363da8-f0f9-410c-84a0-ad58321b4739
# ╠═e2944b0f-7cd0-4356-ae72-c0ab63a61301
# ╠═86cb37a0-e006-4b74-8ccb-99f149d78fe1
# ╠═5bb112bf-6f74-403b-bef3-2251224d6b81
# ╠═313afa4b-4a9c-42c8-a1aa-eb66ea6462d7
# ╠═2eb18ebe-5a47-4ad8-a7d4-44ef58e9f657
# ╠═f15b56a3-a334-4dd3-beee-34e08e4994d6
# ╠═cf3c8858-f6ff-4c11-9869-0dd04c34cf25
# ╠═b531fda7-c102-4ee0-8214-5f523caa2ffe
# ╠═d332c4a3-4e1d-4b2f-bcc6-977df4555181
# ╠═652bfb25-6e92-4f26-9a2e-6242449f00af
# ╠═2a6c54b3-425c-43a5-8b72-53f78d31b13d
# ╠═1ca887db-7027-4ae9-bc57-6f3b4cee3c16
# ╠═46bf3560-b7c2-4a04-bb6c-c878c199edb8
# ╠═3752bcd2-7e98-4a3d-b2f5-0584417513c8
# ╠═a805b547-a9cb-44c7-8a43-aecada6693bb
# ╠═23959308-43b0-49a7-a89d-027c6377f792
# ╠═ed9c8f93-c520-4712-af5b-c372fef4c6a8
# ╠═e299f27a-576f-44b3-b49b-9ad6728cb446
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
