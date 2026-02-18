import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

#function for calculating fit quality
def calculate_fit_quality(data, params, dist_function):
    hist_values, bin_edges = np.histogram(data, density=True, bins=10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    curve_values = dist_function(bin_centers, *params)
    mean_error = np.mean(np.abs(hist_values - curve_values))
    max_error = np.max(np.abs(hist_values - curve_values))
    return mean_error, max_error

st.title("📊 Distribution Fitting Tool")
st.caption("NE111 Programming Project")

st.subheader("📥 Data Input")
st.caption("Upload your CSV file or manually input your data")
uploaded_file = st.file_uploader("Upload your CSV file", type ="csv")
txt = st.text_area("Enter numbers separated by commas")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    final_data = df.iloc[:, 0].tolist()  # Get first column as list

elif txt:
   data_clean = txt.replace(',',' ').split()
   final_data = [float(x) for x in data_clean]
else:
   st.write("Please enter data using either method")
   final_data = []

st.write("Final data:", str(final_data))
st.write("Number of data points:", len(final_data))

#for testing
if final_data:
    st.success(f"✅ Successfully loaded {len(final_data)} data points")

st.divider()
st.subheader("🎯 Distribution Fitting")
options = st.selectbox("What distribution would you like?",
                       ["Normal", "Gamma", "Weibull", "Exponential","Uniform",
                        "Logistic", "Lognormal", "Beta", "Chi-squared", "Rayleigh"]
)
st.write("You selected",options)

# manual fitting toggle
manual_mode = st.checkbox("🚦 Manual Fitting Mode")

#NORMAL 
if final_data and options == "Normal":
    params = stats.norm.fit(final_data)
    st.write("Normal distribution parameters:", params)
    st.write("Mean:", params[0])
    st.write("Standard deviation:", params[1])

    # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.norm.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Normal":
     data_min = float(min(final_data))
     data_max = float(max(final_data))
     data_range = data_max - data_min
    
     mean_slider = st.slider("Mean", data_min, data_max, float(np.mean(final_data)))
     std_slider = st.slider("Standard Deviation", 0.1, data_range, float(np.std(final_data)))
    
     params = (mean_slider, std_slider)
#GAMMA 
if final_data and options == "Gamma":
    mean = np.mean(final_data) #Using method of moments
    std = np.std(final_data)
    shape = (mean / std) ** 2
    scale = (std ** 2) / mean
    params = (shape, 0, scale)  # Gamma uses (a, loc, scale)
    st.write("Gamma distribution parameters:", params)
    st.write("Shape parameter:", params[0])
    st.write("Scale parameter:", params[2])  # Note: scale is the third parameter

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.gamma.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Gamma":
     data_max = float(max(final_data))
    
     shape_slider = st.slider("Shape parameter", 0.1, 20.0, 5.0)
     scale_slider = st.slider("Scale parameter", 0.1, data_max, float(np.std(final_data)))
    
     params = (shape_slider, 0, scale_slider)

#WEIBULL 
if final_data and options == "Weibull":
    st.write("Data being used:", final_data)
    mean = np.mean(final_data) #Using method of moments
    std = np.std(final_data)
    shape = 1.2  # Reasonable starting shape
    scale = mean / math.gamma(1 + 1/shape)
    params = (shape, min(final_data), scale)  # (shape, loc, scale)
    st.write("Weibull parameters (method of moments):", params)
    st.write("Shape parameter (c):", params[0])
    st.write("Scale parameter (λ):", params[2])

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.weibull_min.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Weibull":
     data_min = float(min(final_data))
     data_max = float(max(final_data))
    
     shape_slider = st.slider("Shape (c)", 0.1, 10.0, 1.5)
     scale_slider = st.slider("Scale (λ)", 0.1, data_max, float(np.mean(final_data)))
    
     params = (shape_slider, data_min, scale_slider)

#EXPONENTIAL
if final_data and options == "Exponential":
    params = stats.expon.fit(final_data)
    st.write("Exponential distribution parameters:", params)
    st.write("Scale parameter (1/λ):", params [1])

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.expon.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")
    
    #making the sliders
    if manual_mode and options == "Exponential":
     scale = st.slider("Scale (1/λ)", 0.1, float(max(final_data)), float(np.mean(final_data)))
     params = (0, scale)

#UNIFORM
if final_data and options == "Uniform":
    params = stats.uniform.fit(final_data)
    st.write("Uniform distribution parameters:", params)
    st.write("Location parameter:", params[0])
    st.write("Scale parameter:", params[1])

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.uniform.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Uniform":
     loc = st.slider("Minimum", float(min(final_data)), float(max(final_data)), float(min(final_data)))
     scale = st.slider("Range", 0.1, float(max(final_data)-min(final_data)), float(max(final_data)-min(final_data)))
     params = (loc, scale)

#LOGISTIC
if final_data and options == "Logistic":
    params = stats.logistic.fit(final_data)
    st.write("Logistic distribution parameters:", params)
    st.write("Location parameter:", params[0])
    st.write("Scale parameter:", params[1])

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.logistic.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Logistic":
     loc = st.slider("Location", float(min(final_data)), float(max(final_data)), float(np.mean(final_data)))
     scale = st.slider("Scale", 0.1, float(max(final_data)-min(final_data)), float(np.std(final_data)))
     params = (loc, scale)

#LOGNORMAL
if final_data and options == "Lognormal":
    mean = np.mean(final_data) #Using method of moments
    std = np.std(final_data)
    shape = np.sqrt(np.log(1 + (std/mean)**2))
    scale = np.log(mean) - 0.5 * shape**2
    params = (shape, 0, np.exp(scale))  # Lognormal uses (s, loc, scale)
    st.write("Lognormal parameters (method of moments):", params)
    st.write("Shape parameter (s):", params[0])
    st.write("Scale parameter:", params[2])

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.lognorm.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Lognormal":
     shape = st.slider("Shape (s)", 0.1, 5.0, 1.0)
     scale = st.slider("Scale", 0.1, float(max(final_data)), float(np.mean(final_data)))
     params = (shape, 0, scale)

#BETA
if final_data and options == "Beta":
    # Beta needs values between 0-1
    min_val = min(final_data)
    max_val = max(final_data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in final_data]
    st.write("Normalized data:", normalized_data) #Method of moments instead of of .fit()
    mean = np.mean(normalized_data)
    variance = np.var(normalized_data)
    alpha = mean * ((mean * (1 - mean) / variance) - 1)
    beta = (1 - mean) * ((mean * (1 - mean) / variance) - 1) #Method of moments for beta
    params = (alpha, beta)
    st.write("Beta parameters (method of moments):", params)

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.beta.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Beta":
     alpha = st.slider("Alpha (a)", 0.1, 10.0, 2.0) #data already normalised
     beta = st.slider("Beta (b)", 0.1, 10.0, 2.0)
     params = (alpha, beta)

#CHI-SQUARED
if final_data and options == "Chi-squared":
    mean = np.mean(final_data) #Using method of moments
    df = max(mean, 1) #degrees of freedom ≈ mean
    params = (df,)# Chi-squared only has one parameter (degrees of freedom)
    st.write("Chi-squared distribution parameters:", params)
    st.write("Degrees of freedom:", params[0])

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.chi2.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Chi-squared":
     df = st.slider("Degrees of freedom", 1, 50, int(np.mean(final_data)))
     params = (df,)

#RAYLEIGH
if final_data and options == "Rayleigh":
    params = stats.rayleigh.fit(final_data)
    st.write("Rayleigh distribution parameters:", params)
    st.write("Location parameter:", params[0])
    st.write("Scale parameter:", params[1])

     # Calculate fit quality metrics 
    mean_error, max_error = calculate_fit_quality(final_data, params, stats.rayleigh.pdf)
    st.write("**Fit Quality Metrics:**")
    st.write(f"📊 Mean Error: {mean_error:.4f}") 
    st.write(f"📈 Max Error: {max_error:.4f}")

    #making the sliders
    if manual_mode and options == "Rayleigh":
     scale = st.slider("Scale", 0.1, float(max(final_data)), float(np.mean(final_data)))
     params = (0, scale)

st.divider()
 # HISTOGRAM
if final_data:
    st.header("📉 Histogram visualisation")
    fig,ax = plt.subplots() # creating a blank canvas for graph
    ax.hist(final_data, density=True, alpha=0.7, label='Data') # drawing the histograms

    #if normal is added
    if options == "Normal":
        x = np.linspace(min(final_data), max(final_data), 100)
        pdf = stats.norm.pdf(x, *params)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Normal')
    
    #if gamma is added
    if options == "Gamma":
        x = np.linspace(min(final_data), max(final_data), 100)
        pdf = stats.gamma.pdf(x, *params)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Gamma')

    #if weibull is added
    if options == "Weibull":
     x = np.linspace(min(final_data), max(final_data), 100)
     pdf = stats.weibull_min.pdf(x, *params)
     ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Weibull')
    
    #if exponential is added
    if options == "Exponential":
        x = np.linspace(min(final_data), max(final_data), 100)
        pdf = stats.expon.pdf(x, *params)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Exponential')
    
    #if uniform is added
    if options == "Uniform":
        x = np.linspace(min(final_data), max(final_data), 100)
        pdf = stats.uniform.pdf(x, *params)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Uniform')

    #if Logistic is added
    if options == "Logistic":
        x = np.linspace(min(final_data), max(final_data), 100)
        pdf = stats.logistic.pdf(x, *params)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Logistic')
    
    #if lognormal is added
    if options == "Lognormal":
        x = np.linspace(min(final_data), max(final_data), 100)
        pdf = stats.lognorm.pdf(x, *params)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Lognormal')
    
     #if beta is added
    if options == "Beta":
     x = np.linspace(0, 1, 100) #Beta is defined 0-1
     pdf = stats.beta.pdf(x, *params)
     min_val = min(final_data)
     max_val = max(final_data)
     x_original = [x_val * (max_val - min_val) + min_val for x_val in x]
     ax.plot(x_original, pdf, 'r-', linewidth=2, label='Fitted Beta')

    #if chi-squared is added
    if options == "Chi-squared":
     x = np.linspace(0, max(final_data)*2, 100) 
     pdf = stats.chi2.pdf(x, *params)
     ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Chi-squared')

    #if Rayleigh is added
    if options == "Rayleigh":
        x = np.linspace(min(final_data), max(final_data), 100)
        pdf = stats.rayleigh.pdf(x, *params)
        ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted Rayleigh')

    ax.set_title("Histogram of your data")
    ax.set_xlabel("Values")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig) #Show the matplotlib graph in my streamlit app



