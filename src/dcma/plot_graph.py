import pandas as pd
from plotnine import (ggplot, ggtitle, aes, geom_col, theme_dark, element_text, theme,
                      geom_histogram, geom_point, geom_smooth
                    )
import numpy as np
from scipy.stats import iqr
import plotly.graph_objects as go


def plot_histogram(data: pd.DataFrame, variable_to_plot: str, 
                   title: str = None, bins_method: str = 'freedman-Diaconis'
                   ):
    data = data.dropna(subset=variable_to_plot).copy()
    
    # by default Freedman–Diaconis rule is computed to determine 
    #optimal number of bins for histogram
    
    if bins_method == 'freedman-Diaconis':
        h = (2 * iqr(np.array(data[variable_to_plot].dropna().values))) / (data[variable_to_plot].shape[0]#len(data[variable_to_plot])
              **(1/3)
            )
        if h <= 0:
            nbins = 30
        else:
            nbins = (data[variable_to_plot].max() - data[variable_to_plot].min()) / h
            nbins = round(nbins, 1)
    else:
        nbins = 30
    if title is None:
        title = f"Distribution of {variable_to_plot}"
    histogram = (ggplot(data, aes(x=variable_to_plot))
                + geom_histogram(bins=nbins)
                + ggtitle(title) 
            )
    histogram.show()


def plot_scatterplot(data: pd.DataFrame,
                      x_colname: str,
                      y_colname: str = 'hits'
                      ):
    """ Scatterplot to visualize relationship between two variables. 
    Args:
        data (pd.DataFrame): Data which contains variables to plot
        
        y_colname (str): column name (variable) to plot on y-axis
        x_colname (str): column name (variable) to plot on x-axis
    """
    scatter_graph = (ggplot(data=data, mapping=aes(y=y_colname, x=x_colname)) 
                            + geom_point() + geom_smooth(method='lm')
                            + ggtitle(f'Scatter plot to visualize relationship between {y_colname} and {x_colname}')
                    )
    scatter_graph.show()
    

def plot_cpa_by_variable(data, x_colname: str,
                        y_colname: str, 
                        fill_colname: str
                        ) -> ggplot:
    """Compute CPA by variable specified

    Args:
        data (pd.DataFrame): Data to analyze
        x_colname (str): X axis variable
        y_colname (str): y axis variable
        fill_colname (str): Variable to use for color

    Returns:
        _type_: GGPLOT
    """
    fig = (ggplot(data, aes(x=x_colname, y=y_colname, fill=fill_colname))
            + geom_col(stat='identity', position='dodge')
            + theme_dark()
            + ggtitle(f'Variation in CPA by {x_colname}')
            + theme(axis_text_x=element_text(angle=45, hjust=1),
                    legend_position='bottom'
                    )
            )
    fig.show()


def visualize_cv_metrics(cv_results, metrics = ['accuracy', 'precision', 'recall', 'f1']):
    """
    Plots grouped bar charts of CV metrics (train vs test splits) with a dark background,
    and annotates each bar with its performance value.
    
    Parameters:
      cv_results (dict): Dictionary containing CV results with keys:
        ['train_accuracy', 'test_accuracy',
         'train_precision', 'test_precision',
         'train_recall', 'test_recall',
         'train_f1', 'test_f1']
         Each key should map to a list or array of scores.
    """
    

    
    train_means, test_means = [], []
    train_stds, test_stds = [], []

    for m in metrics:
        train_key = 'train_' + m
        test_key = 'test_' + m

        if train_key not in cv_results or test_key not in cv_results:
            raise KeyError(f"Missing keys for metric {m} in cv_results.")

        train_vals = np.array(cv_results[train_key])
        test_vals = np.array(cv_results[test_key])
        train_means.append(np.mean(train_vals))
        train_stds.append(np.std(train_vals))
        test_means.append(np.mean(test_vals))
        test_stds.append(np.std(test_vals))
    
    train_text = [f"{mean:.3f}" for mean in train_means]
    test_text = [f"{mean:.3f}" for mean in test_means]
    
    fig = go.Figure(data=[
        go.Bar(
            name='Train',
            x=metrics,
            y=train_means,
            text=train_text,
            textposition='outside',
            error_y=dict(type='data', array=train_stds, visible=True),
            marker_color='rgb(55, 83, 109)'
        ),
        go.Bar(
            name='Test',
            x=metrics,
            y=test_means,
            text=test_text,
            textposition='outside',
            error_y=dict(type='data', array=test_stds, visible=True),
            marker_color='rgb(26, 118, 255)'
        )
    ])
    
    fig.update_layout(
        template="plotly_dark",
        title="CV Metrics: Train vs Test Performance",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode='group'
    )
    fig.show()

