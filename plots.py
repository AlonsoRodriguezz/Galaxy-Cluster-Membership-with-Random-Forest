import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# astropy
from astropy.cosmology import LambdaCDM
from astropy.constants import c
from astropy import units as u

# sklearn
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, precision_score, recall_score

# =============================================================================
# CONFIGURATION
# =============================================================================

# Colors
c_tn = '#b0b0b0'         # colors for confusion matrix flags
c_tp = 'green' 
c_fn = 'red' 
c_fp = 'gold'  
c_comp = '#004c6d'   # colors for purity and completeness
c_pur = '#d62728'

# Cosmology
OMegaM = 0.3089
OmegaL = 0.6911 
h = 0.6774
cosmo = LambdaCDM(H0 = h*100, Om0 = OMegaM, Ode0 = OmegaL)
c_km_s = c.to(u.km / u.s).value    # c constant

# ==============================================================================

def plot_learning_curve(model, X, y, groups, ylim = None, cv = None, n_jobs = -1, train_sizes = np.linspace(.1, 1.0, 5), scoring = 'f1', model_name = None):
    """
    Plots a learning curve (metric vs. training examples).
        
    Args:
        model (sklearn.base.BaseEstimator): Model to use for the fit.
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        groups (array-like): groups for the Cross-Validation (e.g. Leave-One-Group-Out ). 
        ylim (tuple, optional): Defines minimum and maximum y values (ymin, ymax) in the plot.
        cv (int or cross-validation generator): Determines the Cross-Validation strategy.
        n_jobs (int, optional): Number of jobs to run in parallel.
        train_sizes (array-like): Relative or absolute numbers of training examples.
        scoring (str): Target metric to evaluate (F1- Score by default). 
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    c_train = 'red'           # colors
    c_test  = '#1E90FF'

    # Learning curve data
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, groups = groups, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes, scoring = scoring)
    train_mean, train_std = np.mean(train_scores, axis = 1), np.std(train_scores, axis = 1)
    test_mean, test_std = np.mean(test_scores, axis = 1), np.std(test_scores, axis = 1)
    
    fig, ax = plt.subplots(figsize = (10, 6))
    if ylim is not None:
        plt.ylim(*ylim)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha = 0.1, color = c_train)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha = 0.1, color = c_test)
    ax.plot(train_sizes, train_mean, 'o-', color = c_train, label = 'Training score', linewidth = 2, markersize = 6)
    ax.plot(train_sizes, test_mean, 'o-', color = c_test, label = 'Cross-Validation score',linewidth = 2, markersize = 6)

    def k_formatter(x, pos):           # Formatter for x-axis (e,g,, 1000 -> 1k)
        return f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'
        
    ax.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax.set_xlabel('Training examples', fontsize = 14, labelpad = 20)
    ax.set_ylabel('F1-Score', fontsize = 14)    # can change if you switch validation metric
    ax.legend(loc='best', fontsize = 12, frameon = True, fancybox = True, framealpha = 0.9)
    
    if model_name:
        plt.savefig(f'{model_name}_LC.pdf', dpi = 250, bbox_inches = 'tight')
    plt.show()
    plt.close()

############################################################################
############################################################################
############################################################################

def plot_confusion_matrix(cm, classes = ['Interloper', 'Member'], normalize = True, title =' Normalized Confusion Matrix', model_name = None, cmap= 'Blues'):
    """
    Prints and plots a confusion matrix with or without normalization.
        
    Args:
        cm (np.ndarray): Confusion matrix array.
        classes (list): List of class names for the axis labels.
        normalize(bool, optional): Wether to normalize the confusion matrix or not. 
        title (str, optional): Title of the plot.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
        cmap (str or Colormap, optional): Colormap for the heatmap.
    """
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        cm_norm = cm   #  Use raw count 
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    plt.figure(figsize = (7,6))
    
    im = plt.imshow(cm_norm if normalize else cm, interpolation = 'nearest', cmap = cmap)
    
    if normalize:
        im.set_clim(0, 1)      # Ensure range is 0-1 for normalized
    
    cbar = plt.colorbar(im)
    cbar.set_label('Rate' if normalize else 'Count', rotation = 270, labelpad = 20, fontsize = 14)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 14)
    plt.yticks(tick_marks, classes, fontsize = 14, rotation = 90, va = 'center')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):        
        
        count_val = cm[i, j]                # Absolute value  (Counts)
        if normalize: 
            rate_val = cm_norm[i, j]   # Rate
            text_top = f'{rate_val:.2f}'
            text_color_check = rate_val
        else:
            text_top = f'{count_val}'
            text_color_check = count_val / cm.max()
        text_color = 'white' if text_color_check > 0.5 else 'black'   # Text color: white if background is dark/strong
            
        plt.text(j, i , text_top, ha = 'center', va = 'center', color = text_color, fontsize = 20, fontweight = 'bold')   # rate
        if normalize:
            plt.text(j, i+0.1, f'({count_val})', ha = 'center', va = 'center', color = text_color, fontsize = 13)                # count
        
    plt.tight_layout()
    plt.ylabel('True label', fontsize = 14, labelpad = 20)
    plt.xlabel('Predicted label', fontsize = 14, labelpad = 20)
    
    if model_name:
        plt.savefig(f'{model_name}_CM.pdf', bbox_inches = 'tight', dpi = 250)    
    plt.show()
    plt.close()

############################################################################
############################################################################
############################################################################

def plot_feature_importances(model, X, y, model_name=None):
    """
    Plots the Random Forest Feature Importance (Mean Decrease Impurity).
        
    Args:
        model (sklearn.base.BaseEstimator): Model to use for the fit.
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    
    feature_importances = model.feature_importances_
    importance = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by = 'Importance', ascending = True)

    names = {'V_norm': r'$V_{pec}/\sigma_{200}$',       # labels for ticks
                     'R_norm': r'$r_{proj}/r_{200}$',
                     'log_local_density': r'log $\Sigma_{10}$', 
                     'Mvir': r'$M_{200}$',
                     'r_mag': r'$m_{r}$'}

    importance['Feature'] = importance['Feature'].map(names)
    
    fig, ax = plt.subplots(figsize = (10, 6))

    # Normalize colormap
    norm = mcolors.Normalize(vmin = importance['Importance'].min(), vmax = importance['Importance'].max())
    cmap = plt.get_cmap('Blues')
    bar_colors = cmap(norm(importance['Importance'].values))

    bars = ax.barh(importance['Feature'], importance['Importance'], color = bar_colors, edgecolor = 'none', height = 0.7)

    max_val = importance['Importance'].max()
    ax.set_xlim(0, max_val * 1.10)

     # Add label to each bar
    for rect, value in zip(bars, importance['Importance']):
        width = rect.get_width()
        font_weight = 'bold' if value == max_val else 'normal'
            
        ax.text(width + (max_val * 0.01), rect.get_y() + rect.get_height()/2, f'{value:.3f}', va = 'center', ha = 'left', 
                    fontsize = 11, fontweight = font_weight, color = 'black')
    
    ax.set_xlabel('Feature Importance', fontsize = 14)
    ax.tick_params(axis = 'y', labelsize = 13, length = 0)

    plt.tight_layout()
    if model_name:
        plt.savefig(f'{model_name}_FeatureImportance.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()

############################################################################
############################################################################
############################################################################

def plot_phase_space(results,model_name=None):
    """
    Plots a Phase Space Diagram color-coded witht the confusion matrix flags (tp, tn, fp, fn).
    
    Args:
        results (pd.DataFrame): Dataframe with summarized results and predictions of the model.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    fig, ax = plt.subplots(figsize = (10, 8))

    mask_TP = results['class'] == 'TP'
    mask_FP = results['class'] == 'FP'
    mask_TN = results['class'] == 'TN'
    mask_FN = results['class'] == 'FN'
    
    ax.scatter(results.loc[mask_TN, 'R_norm'], results.loc[mask_TN, 'V_norm'], s = 2, alpha = 0.3, color = c_tn, label='TN', zorder=1)
    ax.scatter(results.loc[mask_TP,  'R_norm'], results.loc[mask_TP, 'V_norm'],  s = 5, alpha = 0.8, color = c_tp, label='TP', zorder=2)
    ax.scatter(results.loc[mask_FN, 'R_norm'], results.loc[mask_FN, 'V_norm'], s = 4, alpha = 0.5, color = c_fn, label='FN', zorder=3)
    ax.scatter(results.loc[mask_FP,  'R_norm'], results.loc[mask_FP, 'V_norm'],  s = 4, alpha = 0.5, color = c_fp, label='FP', zorder=4)    

    kde_args = {'ax': ax, 'levels': 5, 'thresh': 0.3, 'linewidths': 2.5}       # density contours
    sns.kdeplot(x = results.loc[mask_TP, 'R_norm'], y = results.loc[mask_TP, 'V_norm'],  color = 'darkgreen', zorder = 7, **kde_args)
    sns.kdeplot(x = results.loc[mask_FN,'R_norm'], y = results.loc[mask_FN, 'V_norm'], color = 'darkred', zorder = 5, **kde_args)
    sns.kdeplot(x = results.loc[mask_FP, 'R_norm'], y = results.loc[mask_FP, 'V_norm'], color = 'darkorange', zorder = 6, **kde_args)
    
    ax.set_xlabel(r'$r_{proj} / r_{200}$', fontsize=22)
    ax.set_ylabel(r'$\Delta v / \sigma_{200}$', fontsize=22)
    ax.set_xlim(0, 5.2)
    ax.set_ylim(-4, 4)

    legend_elements = [   # custom legend
        Line2D([0], [0], color = c_tn, marker = 'o', linestyle = 'None', markersize = 5, label = 'TN', alpha = 0.5),
        Line2D([0], [0], color = c_tp, marker = 'o', markersize = 5, label = 'TP', linestyle = '-'),
        Line2D([0], [0], color = c_fn, marker = 'o', markersize = 5, label = 'FN', linestyle = '-'),
        Line2D([0], [0], color = c_fp, marker = 'o', markersize = 5, label = 'FP', linestyle = '-')]

    ax.legend(handles = legend_elements, fontsize = 13, loc = 1, framealpha = 0.9)
    plt.tight_layout()
    if model_name:
        plt.savefig(f'{model_name}_PPS.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()

############################################################################
############################################################################
############################################################################

def plot_purity_and_completeness(y, y_pred, results, model_name=None, n_bins=25):
    """
    Plots purity and completeness as a function of the cluster-centric distance in bins. 
    Includes boxplots for each bin to visualize the variance between clusters.

    Args:
        y (array-like): True labels.
        y_pred (array-like): Predicted labels.
        results (pd.DataFrame): Dataframe with the summarized results and predictions for the model.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
        n_bins (int): Number of radial bins to use. Defaults to 25.
    """
    # bin configuration
    r_min, r_max = 0, 5
    bins = np.linspace(r_min, r_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    width_box = bin_width * 0.25

    global_purity = precision_score(y, y_pred)      # global metrics (over full confusion matrix)
    global_comp = recall_score(y, y_pred)
    
    tp_global, _ = np.histogram(results.loc[results['class'] == 'TP', 'R_norm'], bins = bins)   # global metrics per bin
    fp_global, _ = np.histogram(results.loc[results['class'] == 'FP', 'R_norm'], bins = bins)
    fn_global, _ = np.histogram(results.loc[results['class'] == 'FN', 'R_norm'], bins = bins)

    purity_bin = np.divide(tp_global, tp_global + fp_global, out = np.full_like(tp_global, np.nan, dtype = float), 
                           where = (tp_global+fp_global) != 0)   # safe division to handle empty bins
    completeness_bin = np.divide(tp_global, tp_global + fn_global, out = np.full_like(tp_global, np.nan, dtype = float), 
                                 where = (tp_global+fn_global) != 0)

##################################################################################
    print(f"{'Bin Center':<12} | {'TP':<6} | {'FN':<6} | {'Total':<10} | {'Completeness':<12}") #stats table
    print('-' * 55)
    for i in range(len(bin_centers)):
        total_true = tp_global[i] + fn_global[i]
        comp_val = completeness_bin[i]
        print(f'{bin_centers[i]:<12.2f} | {tp_global[i]:<6} | {fn_global[i]:<6} | {total_true:<10} | {comp_val:.4f}')
##################################################################################
    
    # Metrics per cluster (boxplots)
    clusters = results['mock_id'].unique()
    matrix_pur = np.zeros((len(clusters), n_bins))   # initialize list of lists
    matrix_com = np.zeros((len(clusters), n_bins))
    matrix_pur[:] = np.nan
    matrix_com[:] = np.nan

    for i, cl_id in enumerate(clusters):
        sub = results[results['mock_id'] == cl_id]
        tp, _ = np.histogram(sub.loc[sub['class'] == 'TP', 'R_norm'], bins = bins)
        fp, _ = np.histogram(sub.loc[sub['class'] == 'FP', 'R_norm'], bins = bins)
        fn, _ = np.histogram(sub.loc[sub['class'] == 'FN', 'R_norm'], bins = bins)
        denom_p = tp + fp
        denom_c = tp + fn
        matrix_pur[i, :] = np.divide(tp, denom_p, out = np.full_like(tp, np.nan, dtype = float), where = denom_p != 0)
        matrix_com[i, :] = np.divide(tp, denom_c, out = np.full_like(tp, np.nan, dtype = float), where = denom_c != 0)

    # clean NaNs for boxplots
    data_pur = []
    data_com = []
    for i in range(n_bins):
        col_p = matrix_pur[:, i]
        col_c = matrix_com[:, i]
        data_pur.append(col_p[~np.isnan(col_p)])
        data_com.append(col_c[~np.isnan(col_c)])

    # PLOT
    fig, ax = plt.subplots(figsize=(12, 6))

    # Global metric lines 
    ax.plot(bin_centers, completeness_bin, color = c_comp, marker = 'o', markersize = 5, linestyle = '-', linewidth = 2,   # global completeness
                 label = f'Completeness (Global: {global_comp:.2f})', zorder = 10)

    ax.plot(bin_centers, purity_bin, color = c_pur, marker = 's', markersize = 5, linestyle = '--', linewidth = 2,                  # global purity
                 label = f'Purity (Global: {global_purity:.2f})', zorder = 10)

    # boxplots configuration
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color = color, linewidth = 1)
        plt.setp(bp['whiskers'], color = color, linewidth = 1)
        plt.setp(bp['caps'], color = color, linewidth = 1)
        plt.setp(bp['medians'], color = color, linewidth = 1.5)

    # Boxplots, no outliers
    bp_com = ax.boxplot(data_com, positions = bin_centers, widths = width_box, patch_artist = True, showfliers = False, zorder = 2)     # completeness
    bp_pur = ax.boxplot(data_pur, positions = bin_centers, widths = width_box, patch_artist = True, showfliers = False, zorder = 2)        # purity
    
    set_box_color(bp_com, c_comp)    # boxplots colors
    set_box_color(bp_pur, c_pur)
    for patch in bp_com['boxes']:
        patch.set_facecolor(c_comp)
        patch.set_alpha(0.4)
    for patch in bp_pur['boxes']:
        patch.set_facecolor(c_pur)
        patch.set_alpha(0.4)
    
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r'$r_{proj} / r_{200}$', fontsize = 17)
    ax.set_ylabel('Rate', fontsize = 15)
    ax.set_xticks( [0, 1, 2, 3, 4, 5])
    ax.set_xticklabels( [0, 1, 2, 3, 4, 5], fontsize = 12)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    handles = [plt.Line2D([0], [0], color = c_comp, marker = 'o', label = f'Completeness (Global: {global_comp:.2f})'),           # custom legend
               plt.Line2D([0], [0], color = c_pur, marker = 's', linestyle = '--', label = f'Purity (Global: {global_purity:.2f})'),
               Patch(facecolor = 'gray', edgecolor = 'gray', alpha = 0.4, label = 'Cluster Variability (IQR)')]

    ax.legend(handles = handles, fontsize = 12, loc = 'lower left', framealpha = 0.75, fancybox = True)
    plt.tight_layout()
    
    if model_name:
        plt.savefig(f'{model_name}_purandcomp.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()

############################################################################
############################################################################
############################################################################

def plot_roc_curve(y_true, y_prob, thresh, model_name=None):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and highlights the selected threshold.
        
    Args:
        y_true (array-like): True labels.
        y_prob (array-like): Probability estimates of the positive class.
        thresh (float): Decision threshold chosen for predictions.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    # Find best F1-Score based on Precision-Recall 
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    fscore = (2 * precision * recall) / (precision + recall)
    ix_f1 = np.argmax(fscore)
    best_thresh_f1 = thresholds_pr[ix_f1]

    # Find indices for plotting points
    idx_f1 = np.argmin(np.abs(thresholds_roc - best_thresh_f1))  # max f1 index
    idx_chosen = np.argmin(np.abs(thresholds_roc - thresh))         # chosen threshold index

    fig, ax = plt.subplots(figsize = (8, 7))

    ax.plot(fpr, tpr, color = '#004c6d', lw = 2, label = f'AUC = {auc_score:.2f}', zorder = 1) # ROC
    ax.fill_between(fpr, tpr, alpha = 0.1, color = '#1E90FF', zorder = 2)

    ax.scatter(fpr[idx_chosen], tpr[idx_chosen], s = 100, facecolors = 'white', edgecolors = '#d62728', lw = 2, zorder = 4)
    ax.scatter(fpr[idx_chosen], tpr[idx_chosen], s = 20, color = '#d62728', label = f'Threshold used = {thresh}', zorder = 5)
    ax.scatter(fpr[idx_f1], tpr[idx_f1], marker = 'x', color = 'black', s = 100, label = f'Max F1 (Th={best_thresh_f1:.2f})', zorder = 6)
    ax.plot([0, 1], [0, 1], linestyle = '--', lw = 1.5, color = 'gray', label = 'Random Guessing')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate', fontsize = 14)
    ax.set_ylabel('True Positive Rate', fontsize = 14)
    ax.legend(loc = 'lower right', frameon = True, fontsize = 11)
    plt.tight_layout()

    if model_name:
        plt.savefig(f'{model_name}_ROC.pdf', dpi = 300, bbox_inches = 'tight')
    plt.show()
    plt.close()

############################################################################
############################################################################
############################################################################

def plot_3d_cm_per_cluster(dff, cluster_catalogue, results, model_name):
    """
    Generates 3D plots color-coded with the confusion matrix flags (tp, tn, fp, fn) for each cluster.
    
    Note: This function creates a directory and saves multiple PDF files.
    
    Args:
        dff (pd.DataFrame): Dataframe with galaxy data (x, y, z, mock_id, etc.).
        cluster_catalogue (pd.DataFrame): Cluster catalog with cluster data (RA, DEC, R200, etc.).
        results (pd.DataFrame): Dataframe with the summarized results and predictions for the model.
        model_name (str): Name of the model, used for directory and file names.
    """
        
    df_working = dff.copy()
    df_working['class_pred'] = results['class'].values

    output_dir = f'3Dplot_ConfusionMatrix_{model_name}'  # directory based on model's name
    os.makedirs(output_dir, exist_ok = True)

    total_clusters = len(cluster_catalogue)

    # Iterate over each mock
    for i, row in cluster_catalogue.iterrows():
        cluster_id = int(row['Cluster'])
        
        # Filter galaxies for this mock
        df_mock = df_working[df_working['mock_id'] == cluster_id].copy()    
        x_cent, y_cent, z_cent = row['X_cent'], row['Y_cent'], row['Z_cent']
        R200 = row['R_200_mpc']

        df_mock['x0'] = df_mock['x'] - x_cent   # center coords
        df_mock['y0'] = df_mock['y'] - y_cent
        df_mock['z0'] = df_mock['z'] - z_cent

        lim = 5 * R200 * 1.3   # limit
        mask_in_box = ((df_mock['x0'].abs() <= lim) & (df_mock['y0'].abs() <= lim) & (df_mock['z0'].abs() <= lim))
        df_plot = df_mock[mask_in_box]

        tp = df_plot[df_plot['class_pred'] == 'TP']  # split by classification
        fp = df_plot[df_plot['class_pred'] == 'FP']
        fn = df_plot[df_plot['class_pred'] == 'FN'] 
        tn = df_plot[df_plot['class_pred'] == 'TN'] 

        # Downsampling TN for visual clarity
        if len(tn) > 5000:
            tn = tn.sample(n = 5000, random_state = 42)

        # PLOT 
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(111, projection = '3d')
        
        ax.scatter(tn['x0'], tn['y0'], tn['z0'], c = c_tn, s = 5, alpha = 0.3, label = f'TN: {len(tn)}', zorder = 1)
        ax.scatter(tp['x0'], tp['y0'], tp['z0'], c = c_tp, s = 10, alpha = 0.6, label = f'TP: {len(tp)}', zorder = 2)
        ax.scatter(fn['x0'], fn['y0'], fn['z0'], c = c_fn, s = 15, marker = '^', alpha = 0.8, label = f'FN: {len(fn)}', zorder = 11)
        ax.scatter(fp['x0'], fp['y0'], fp['z0'], c = c_fp, s = 15, marker = 'x', alpha = 0.9, label = f'FP: {len(fp)}', zorder = 12)

        # Wireframe  sphere (5R_200)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        r_sphere = 5 * R200
        x_sph = r_sphere * np.cos(u) * np.sin(v)
        y_sph = r_sphere * np.sin(u) * np.sin(v)
        z_sph = r_sphere * np.cos(v)
        ax.plot_wireframe(x_sph, y_sph, z_sph, color = 'k', alpha = 0.1, linewidth = 0.5)

        ax.xaxis.pane.fill = False     # style
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.5, 'alpha': 0.3}
        ax.xaxis._axinfo["grid"].update(grid_style)   
        ax.yaxis._axinfo["grid"].update(grid_style)
        ax.zaxis._axinfo["grid"].update(grid_style)

        ax.set_xlabel('x [Mpc]')
        ax.set_ylabel('y [Mpc]')
        ax.set_zlabel('z [Mpc]')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_title(f'Cluster {cluster_id}')
        ax.legend(loc = 'upper right', fontsize = 'small')

        ax.set_box_aspect([1,1,1])
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'Cluster_{cluster_id}_CM.pdf')
        plt.savefig(filename, dpi = 300)
        plt.close(fig) 
    print('Done')

############################################################################
############################################################################
############################################################################

def plot_3d_cm_stacked(dff, cluster_catalogue, results, model_name = None):
    """
    Plots the stacked 3D clusters color-coded with the confusion matrix flags (tp, tn, fp, fn).

    Args:
        dff (pd.DataFrame): Dataframe with galaxy data (x, y, z, mock_id, etc.).
        cluster_catalogue (pd.DataFrame): Cluster catalog with cluster data (RA, DEC, R200, etc.).
        results (pd.DataFrame): Dataframe with summarized results and predictions for the model.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """

    df_working = dff.copy()
    df_working['class_pred'] = results['class'].values
    stack_tp, stack_fp, stack_fn, stack_tn = [], [], [], []
    total_clusters = len(cluster_catalogue)
    
    for i, row in cluster_catalogue.iterrows():    
        cluster_id = int(row['Cluster'])
    
        df_mock = df_working[df_working['mock_id'] == cluster_id].copy()
        
        x_c, y_c, z_c = row['X_cent'], row['Y_cent'], row['Z_cent']
        r200 = row['R_200_mpc']

        df_mock['x_norm'] = (df_mock['x'] - x_c) / r200
        df_mock['y_norm'] = (df_mock['y'] - y_c) / r200
        df_mock['z_norm'] = (df_mock['z'] - z_c) / r200
        
        lim = 6.0
        mask_box = (df_mock['x_norm'].abs() < lim) & (df_mock['y_norm'].abs() < lim) & (df_mock['z_norm'].abs() < lim)
        df_plot = df_mock[mask_box]
        
        stack_tp.append(df_plot[df_plot['class_pred'] == 'TP'][['x_norm','y_norm','z_norm']])
        stack_fp.append(df_plot[df_plot['class_pred'] == 'FP'][['x_norm','y_norm','z_norm']])
        stack_fn.append(df_plot[df_plot['class_pred'] == 'FN'][['x_norm','y_norm','z_norm']])
        
        tn_subset = df_plot[df_plot['class_pred'] == 'TN']
        tn_subset = tn_subset.sample(frac=0.5, random_state=42)  # downsampling TN
        stack_tn.append(tn_subset[['x_norm','y_norm','z_norm']])

    df_tp = pd.concat(stack_tp) if stack_tp else pd.DataFrame()   # concatenate all stacks
    df_fp = pd.concat(stack_fp) if stack_fp else pd.DataFrame()
    df_fn = pd.concat(stack_fn) if stack_fn else pd.DataFrame()
    df_tn = pd.concat(stack_tn) if stack_tn else pd.DataFrame()

    # PLOT
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df_tn['x_norm'], df_tn['y_norm'], df_tn['z_norm'], c = c_tn, s = 2, alpha = 0.4, label = 'TN', zorder = 1, rasterized = False)
    ax.scatter(df_tp['x_norm'], df_tp['y_norm'], df_tp['z_norm'], c = c_tp, s = 2, alpha = 0.3, label = 'TP', zorder = 2, rasterized = False)
    ax.scatter(df_fn['x_norm'], df_fn['y_norm'], df_fn['z_norm'], c = c_fn, s = 5, marker = 'x', alpha = 0.8, label = 'FN', zorder = 10, rasterized = False)
    ax.scatter(df_fp['x_norm'], df_fp['y_norm'], df_fp['z_norm'], c = c_fp, s = 5, marker = '^', alpha = 0.8, label = 'FP', zorder = 11, rasterized = False)

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]

    # Sphere R_200
    x = 1.0 * np.cos(u) * np.sin(v)
    y = 1.0 * np.sin(u) * np.sin(v)
    z = 1.0 * np.cos(v)
    ax.plot_wireframe(x, y, z, color = 'black', alpha = 0.4, linewidth = 1.0, label = '$R_{200}$') 

    # Sphere 5R_200
    x5 = 5.0 * np.cos(u) * np.sin(v)
    y5 = 5.0 * np.sin(u) * np.sin(v)
    z5 = 5.0 * np.cos(v)
    ax.plot_wireframe(x5, y5, z5, color = 'black', alpha = 0.3, linewidth = 1.0, linestyle = '--', label = '$5R_{200}$')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.2, 'alpha': 0.3}
    ax.xaxis._axinfo['grid'].update(grid_style)
    ax.yaxis._axinfo['grid'].update(grid_style)
    ax.zaxis._axinfo['grid'].update(grid_style)

    ax.set_xlabel(r'$x / r_{200}$', fontsize=17, labelpad=10)
    ax.set_ylabel(r'$y / r_{200}$', fontsize=17, labelpad=10)
    ax.set_zlabel(r'$z / r_{200}$', fontsize=17, labelpad=10)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1,1,1])

    leg = plt.legend(loc = 'upper right', fancybox = True, fontsize = 13)
    for lh in leg.legend_handles: 
        if hasattr(lh, '_sizes'):    # make legend markers more visible
            lh._sizes = [30]
            lh.set_alpha(1)

    plt.tight_layout()

    if model_name:
        plt.savefig(f'{model_name}_AllClusters_3D_CM.pdf', dpi = 400, bbox_inches = 'tight')
    plt.show()
    plt.close()

############################################################################
############################################################################
############################################################################

def plot_3d_cm_obs_coords(dff, cluster_catalogue, results, model_name):
    """
    Generates observational 3D space (RA, DEC, z) plots color-coded with the confusion matrix flags 
    (tp, tn, fp, fn) for each cluster.

    Note: This function creates a directory and saves multiple PDF files.

    Args:
        dff (pd.DataFrame): Dataframe with galaxy data (x, y, z, mock_id, etc.).
        cluster_catalogue (pd.DataFrame): Cluster catalog with cluster data (RA, DEC, R200, etc.).
        results (pd.DataFrame): Dataframe with summarized results and predictions for the model.
        model_name (str, optional): If provided, saves the plot (.pdf) with this name.
    """
    
    output_dir = f'3Dplot_ObsCoords_{model_name}'
    os.makedirs(output_dir, exist_ok = True)

    df = dff.copy()
    df['class_pred'] = results['class'].values

    for i, row in cluster_catalogue.iterrows():
        cluster_id = int(row['Cluster'])
        RA_cl, DEC_cl, z_cl = row['RA'], row['Dec'], row['redshift']
        R200_mpc = row['R_200_mpc']
        df_mock = df[df['mock_id'] == cluster_id].copy()
        
        df_mock['dra'] = (df_mock['RA'] - RA_cl)
        df_mock['ddec'] = df_mock['DEC'] - DEC_cl
        df_mock['dz'] = df_mock['redshift_S_1'] - z_cl
        
        da = cosmo.angular_diameter_distance(z_cl).value   # visual limit based on R200
        r5_deg = np.rad2deg((5 * R200_mpc) / da)
        
        hz = cosmo.H(z_cl).value
        r5_dz = (hz * (5 * R200_mpc)) / c_km_s     # 5*R200: dz = (H(z) * dist) / c

        lim_ang = r5_deg * 3.0   # limit for angular coords, zoom out box in order to see FOG
        lim_z = r5_dz * 4.0         # limit for redshift

        mask_in_box = ((df_mock['dra'].abs() <= lim_ang) & (df_mock['ddec'].abs() <= lim_ang) & (df_mock['dz'].abs() <= lim_z))
        df_plot = df_mock[mask_in_box]
    
        tp = df_plot[df_plot['class_pred'] == 'TP']
        fp = df_plot[df_plot['class_pred'] == 'FP']
        fn = df_plot[df_plot['class_pred'] == 'FN'] 
        tn = df_plot[df_plot['class_pred'] == 'TN'] 

        if len(tn) > 4000:
            tn = tn.sample(n = 4000, random_state = 42)    #downsampling of TN if there are too many

        # Plot
        fig = plt.figure(figsize = (9, 9))
        ax = fig.add_subplot(111, projection = '3d')
        
        ax.scatter(tn['dra'], tn['ddec'], tn['dz'], c = c_tn, s = 5, alpha = 0.3, label = f'TN: {len(tn)}', zorder=1)
        ax.scatter(tp['dra'], tp['ddec'], tp['dz'], c = c_tp, s = 8, alpha = 0.6, label = f'TP: {len(tp)}', zorder=2)
        ax.scatter(fn['dra'], fn['ddec'], fn['dz'], c =  c_fn, s = 12, marker = '^', alpha = 0.8, label = f'FN: {len(fn)}', zorder = 11)
        ax.scatter(fp['dra'], fp['ddec'], fp['dz'], c = c_fp,  s = 12, marker = 'x', alpha = 0.9, label = f'FP: {len(fp)}', zorder = 12)
        
        # 5R200 Sphere
        phi, theta = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x_sph = r5_deg * np.cos(phi) * np.sin(theta)
        y_sph = r5_deg * np.sin(phi) * np.sin(theta)
        z_sph = r5_dz * np.cos(theta)
        ax.plot_wireframe(x_sph, y_sph, z_sph, color="black", alpha=0.2, linewidth=0.5)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_facecolor('none')         
        grid_style = {'color': 'gray', 'linestyle': ':', 'linewidth': 0.5, 'alpha': 0.3}
        ax.xaxis._axinfo["grid"].update(grid_style)   
        ax.yaxis._axinfo["grid"].update(grid_style)
        ax.zaxis._axinfo["grid"].update(grid_style)

        ax.set_xlabel(r'$\Delta \alpha$ [deg]', fontsize = 14, labelpad = 10)
        ax.set_ylabel(r'$\Delta \delta$ [deg]', fontsize = 14, labelpad = 10)
        ax.set_zlabel(r'$\Delta z$', fontsize = 14, labelpad = 15)
        ax.tick_params(axis = 'z', pad = 8)
        
        ax.set_xlim(-lim_ang, lim_ang)
        ax.set_ylim(-lim_ang, lim_ang)
        ax.set_zlim(-lim_z, lim_z)
        ax.set_title(f'Mock {cluster_id} Observational Space', fontsize = 16)

        leg = ax.legend(loc = 'upper right', fontsize = 12, frameon = True, facecolor = 'white', framealpha = 0.8)
        for lh in leg.legend_handles:
            lh.set_sizes([50])
            lh.set_alpha(1)

        # perspective
        ax.set_box_aspect([1, 1, 1.333])       # elongated on z since the limits are different, preserves the sphere
        ax.view_init(elev = 15, azim = -60)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'Cluster_{cluster_id}_ObsCM.pdf')
        plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
        plt.close(fig)
    print('Done')

