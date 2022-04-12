import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt


import textwrap
pd.options.display.max_colwidth=800
pd.options.display.max_columns=None

from textos import texto_titulo
st.title(texto_titulo)