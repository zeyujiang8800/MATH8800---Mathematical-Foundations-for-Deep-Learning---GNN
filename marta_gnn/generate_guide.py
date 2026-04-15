"""Generate a PDF presentation guide for the MARTA GNN demo notebook."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "MARTA_GNN_Presentation_Guide.pdf")

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    topMargin=0.75 * inch,
    bottomMargin=0.75 * inch,
    leftMargin=1 * inch,
    rightMargin=1 * inch,
)

styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle(
    "CoverTitle", parent=styles["Title"], fontSize=28, leading=34,
    spaceAfter=12, textColor=HexColor("#1a237e"),
))
styles.add(ParagraphStyle(
    "CoverSubtitle", parent=styles["Title"], fontSize=16, leading=20,
    spaceAfter=6, textColor=HexColor("#424242"),
))
styles.add(ParagraphStyle(
    "SlideTitle", parent=styles["Heading1"], fontSize=20, leading=24,
    spaceBefore=6, spaceAfter=10, textColor=HexColor("#1a237e"),
    borderWidth=0, borderPadding=0,
))
styles.add(ParagraphStyle(
    "SlideSubtitle", parent=styles["Heading2"], fontSize=15, leading=18,
    spaceBefore=10, spaceAfter=6, textColor=HexColor("#283593"),
))
styles.add(ParagraphStyle(
    "Body", parent=styles["BodyText"], fontSize=11, leading=15,
    spaceAfter=8, alignment=TA_JUSTIFY,
))
styles.add(ParagraphStyle(
    "CustomBullet", parent=styles["BodyText"], fontSize=11, leading=15,
    spaceAfter=4, leftIndent=24, bulletIndent=12, alignment=TA_LEFT,
))
styles.add(ParagraphStyle(
    "CustomCode", parent=styles["Code"], fontSize=9, leading=12,
    spaceAfter=6, backColor=HexColor("#f5f5f5"),
    leftIndent=18, rightIndent=18, borderWidth=0.5,
    borderColor=HexColor("#cccccc"), borderPadding=6,
))
styles.add(ParagraphStyle(
    "TalkingPoint", parent=styles["BodyText"], fontSize=10, leading=13,
    spaceAfter=4, leftIndent=24, bulletIndent=12,
    textColor=HexColor("#4527a0"), fontName="Helvetica-Oblique",
))
styles.add(ParagraphStyle(
    "FooterNote", parent=styles["Normal"], fontSize=9, leading=11,
    textColor=HexColor("#757575"), alignment=TA_CENTER,
))

story = []

def slide_title(text):
    story.append(Paragraph(text, styles["SlideTitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=HexColor("#1a237e"),
                             spaceAfter=10))

def subtitle(text):
    story.append(Paragraph(text, styles["SlideSubtitle"]))

def body(text):
    story.append(Paragraph(text, styles["Body"]))

def bullet(text):
    story.append(Paragraph(text, styles["CustomBullet"], bulletText="\u2022"))

def talking_point(text):
    story.append(Paragraph(f"\U0001f4ac  {text}", styles["TalkingPoint"]))

def code(text):
    story.append(Paragraph(text.replace("\n", "<br/>"), styles["CustomCode"]))

def spacer(h=0.15):
    story.append(Spacer(1, h * inch))

def page_break():
    story.append(PageBreak())


# =====================================================================
# COVER PAGE
# =====================================================================
story.append(Spacer(1, 2.5 * inch))
story.append(Paragraph("MARTA GNN", styles["CoverTitle"]))
story.append(Paragraph("Delay Risk Prediction Demo", styles["CoverSubtitle"]))
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph("Presentation Guide", styles["CoverSubtitle"]))
story.append(Spacer(1, 1.5 * inch))
story.append(Paragraph(
    "A walkthrough of the Jupyter notebook demo covering synthetic data generation, "
    "graph neural networks, baseline comparison, evaluation metrics, and automated "
    "hyperparameter tuning.",
    styles["Body"],
))
page_break()

# =====================================================================
# SLIDE 1 – Project Overview
# =====================================================================
slide_title("1. Project Overview")
body(
    "This project applies <b>Graph Neural Networks (GNNs)</b> to predict delay risk "
    "at transit stops in the MARTA (Metropolitan Atlanta Rapid Transit Authority) "
    "network. Each stop is classified as either <b>on-time</b> (class 0) or "
    "<b>at-risk</b> (class 1) based on historical delay patterns."
)
spacer()
subtitle("Why Graphs?")
body(
    "A transit network is naturally a graph: stops are nodes and the routes "
    "connecting consecutive stops are edges. A GNN can exploit this "
    "relational structure &mdash; if neighboring stops along a route are "
    "delayed, the delay is likely to propagate. Traditional models like "
    "MLPs treat each stop independently and miss this spatial signal."
)
spacer()
subtitle("Notebook Outline")
bullet("Set up environment and imports")
bullet("Configure hyperparameters")
bullet("Load and inspect synthetic demo data")
bullet("Visualize delay distributions and stop maps")
bullet("Train a GCN model and an MLP baseline")
bullet("Evaluate with precision, recall, F1, ROC-AUC, MAE/MSE/RMSE")
bullet("Run Optuna hyperparameter search")
bullet("Compare all models side by side")
page_break()

# =====================================================================
# SLIDE 2 – The Demo Data
# =====================================================================
slide_title("2. How the Demo Data Works")
body(
    "The notebook uses <b>entirely synthetic data</b> generated by "
    "<font face='Courier'>mock_data.py</font>. It is <b>not</b> derived from real "
    "MARTA feeds. The codebase does include real-data loaders "
    "(<font face='Courier'>GTFSLoader</font>, <font face='Courier'>RealtimeLoader</font>) "
    "that can download MARTA's actual GTFS static feed and live protobuf streams, "
    "but the demo avoids them so anyone can run it instantly without API keys or "
    "network access."
)
spacer()
subtitle("What Gets Generated")
data_table = [
    ["Component", "How It's Made", "Default Size"],
    ["Stops", "Random lat/lon ~ N(33.75, 0.06) centered on Atlanta", "800"],
    ["Routes", "Numbered routes; mostly bus (type 3), 2 rail (type 1)", "30"],
    ["Trips", "Each trip randomly selects 8-20 stops in order", "200"],
    ["Stop Times", "Scheduled arrival/departure per trip-stop pair", "~2,800"],
    ["Realtime Delays", "Exponential(60s); 40% inflated by Exp(450s)", "~2,800"],
]
t = Table(data_table, colWidths=[1.1*inch, 2.8*inch, 0.9*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a237e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), colors.white]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
spacer()
subtitle("Labels")
body(
    "A stop is labeled <b>at-risk (1)</b> if its <b>median</b> arrival delay across "
    "all realtime observations exceeds the threshold (default 300 seconds = 5 minutes). "
    "With the current settings, roughly 25-30% of stops end up at-risk, ensuring "
    "enough minority-class samples for meaningful training."
)
spacer()
subtitle("Graph Construction")
body(
    "Nodes = stops. An edge connects stop <i>i</i> to stop <i>j</i> if they appear "
    "as consecutive stops on <b>any</b> trip (bidirectional). Edge weight = scheduled "
    "travel time between the two stops. This yields a sparse, realistic transit "
    "topology."
)
spacer(0.1)
talking_point(
    "Mention: The only realistic aspect is Atlanta's geographic center "
    "(33.75\u00b0N, 84.39\u00b0W). Graph topology, schedules, and delays are all "
    "fabricated. For a real deployment, swap <font face='Courier'>use_mock: True</font> "
    "to <font face='Courier'>False</font> and supply MARTA GTFS URLs."
)
page_break()

# =====================================================================
# SLIDE 3 – Why It Runs Fast Without a GPU
# =====================================================================
slide_title("3. Why This Runs Fast Without a GPU")
body(
    "You might expect deep learning to need GPU acceleration, but the demo trains "
    "in a few seconds on CPU alone. Here is why:"
)
spacer()
bullet(
    "<b>Small graph</b> &mdash; 800 nodes and ~5,600 edges fit entirely in RAM. "
    "GPU acceleration helps when you have millions of nodes."
)
bullet(
    "<b>Low-dimensional features</b> &mdash; each node has only 12 features. "
    "Matrix multiplications are tiny."
)
bullet(
    "<b>Shallow model</b> &mdash; 2 GCN layers with 64 hidden units means "
    "~10K parameters. Modern GPUs are designed for models with millions."
)
bullet(
    "<b>Full-batch training</b> &mdash; the entire graph is one batch. There is "
    "zero data-loading overhead; no mini-batch sampling needed."
)
bullet(
    "<b>Early stopping</b> &mdash; training halts after 50 epochs without "
    "improvement, often finishing well before the 200-epoch limit."
)
spacer()
talking_point(
    "Emphasize: GPU would matter for city-scale graphs (100K+ stops) or "
    "when training on continuous live data streams. For a demo this size, "
    "CPU is actually more efficient due to zero GPU transfer overhead."
)
page_break()

# =====================================================================
# SLIDE 4 – Node Features
# =====================================================================
slide_title("4. Node Features (12 per Stop)")
body(
    "Each stop is described by a 12-dimensional feature vector. The first two come "
    "from GTFS static data; the rest are engineered from schedule and realtime feeds."
)
spacer()
feat_table = [
    ["#", "Feature", "Source", "Description"],
    ["0", "lat", "GTFS static", "Stop latitude"],
    ["1", "lon", "GTFS static", "Stop longitude"],
    ["2", "degree", "Graph topology", "Number of edges connected to this stop"],
    ["3", "n_routes", "Schedule", "Distinct bus/rail routes serving the stop"],
    ["4", "n_trips", "Schedule", "Total trips visiting the stop per day"],
    ["5", "avg_headway", "Schedule", "Mean time (sec) between consecutive trips"],
    ["6", "mean_delay", "Realtime", "Average observed arrival delay (sec)"],
    ["7", "std_delay", "Realtime", "Standard deviation of arrival delay"],
    ["8", "max_delay", "Realtime", "Largest observed delay"],
    ["9", "frac_delayed", "Realtime", "Fraction of arrivals exceeding threshold"],
    ["10", "time_bin_sin", "Schedule", "Sin-encoded median departure hour"],
    ["11", "time_bin_cos", "Schedule", "Cos-encoded median departure hour"],
]
t = Table(feat_table, colWidths=[0.3*inch, 1.0*inch, 0.9*inch, 2.9*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a237e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), colors.white]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t)
spacer()
subtitle("Why Cyclical Time Encoding?")
body(
    "Hours are cyclical: 23:00 and 01:00 are only 2 hours apart, but numerically "
    "they look far apart (23 vs 1). By encoding the median departure hour as "
    "sin(2\u03c0 \u00b7 hour/24) and cos(2\u03c0 \u00b7 hour/24), the model "
    "sees a smooth, distance-preserving representation of time of day."
)
page_break()

# =====================================================================
# SLIDE 5 – Arrival Delay Distribution
# =====================================================================
slide_title("5. Distribution of Arrival Delays")
body(
    "The <b>delay distribution histogram</b> (notebook cell 11) shows how many "
    "stop-time observations fall into each delay bucket."
)
spacer()
subtitle("What You'll See")
bullet(
    "A large spike near zero &mdash; most arrivals are close to on-time."
)
bullet(
    "A long right tail &mdash; a meaningful fraction of observations have large "
    "delays (300+ seconds)."
)
bullet(
    "A red vertical line at 300 seconds (5 min) &mdash; this is the threshold. "
    "Observations to the right contribute to the <b>at-risk</b> label."
)
spacer()
subtitle("How Delays Are Generated")
body(
    "60% of observations: Exponential(scale=60s) &mdash; mostly small delays. "
    "40% of observations: Exponential(scale=450s) &mdash; larger delays that push "
    "stops above the threshold. This mixture yields approximately 25-30% of stops "
    "being classified as at-risk, providing a moderately imbalanced dataset."
)
spacer()
talking_point(
    "Point out the threshold line on the histogram. Everything to the right "
    "is what the model is trying to detect."
)
page_break()

# =====================================================================
# SLIDE 6 – Stop Map
# =====================================================================
slide_title("6. The Stop Map")
body(
    "The <b>stop map</b> is a geographic scatter plot where each dot is a transit "
    "stop, positioned by its latitude and longitude (centered on Atlanta)."
)
spacer()
subtitle("Ground Truth Map")
bullet("Green dots = on-time stops (class 0)")
bullet("Red dots = at-risk stops (class 1)")
body(
    "This map appears early in the notebook to give a spatial overview before "
    "any model is trained."
)
spacer()
subtitle("Predicted Stop Map")
body(
    "After training, the same map is drawn using the <b>model's predictions</b> "
    "instead of true labels. Comparing the two maps visually shows where the model "
    "agrees or disagrees with ground truth."
)
spacer()
talking_point(
    "If the predicted map looks nearly identical to the ground truth map, the model "
    "is doing well. Scattered mismatches reveal stops the model finds hard to classify."
)
page_break()

# =====================================================================
# SLIDE 7 – GCN
# =====================================================================
slide_title("7. Graph Convolutional Network (GCN)")
body(
    "A <b>GCN</b> generalizes the concept of convolution from images (grids) to "
    "graphs (irregular structures). At each layer, every node <b>aggregates</b> "
    "features from its neighbors and updates its own representation."
)
spacer()
subtitle("Architecture in This Project")
arch_table = [
    ["Layer", "Details"],
    ["Input", "12 features per node"],
    ["GCNConv 1", "12 \u2192 64, BatchNorm, ReLU, Dropout(0.5)"],
    ["GCNConv 2 (output)", "64 \u2192 2 (raw logits for 2 classes)"],
    ["Activation", "Softmax applied at inference for probabilities"],
]
t = Table(arch_table, colWidths=[1.5*inch, 3.8*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a237e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), colors.white]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
spacer()
subtitle("Key Concepts")
bullet(
    "<b>Message passing</b> &mdash; each node sends its features to neighbors. "
    "The GCNConv layer computes a weighted average of neighbor features, "
    "normalized by node degree."
)
bullet(
    "<b>Skip connections</b> &mdash; when input and output dimensions match, "
    "the model adds the input to the output (residual connection) to help "
    "gradient flow."
)
bullet(
    "<b>BatchNorm</b> &mdash; normalizes activations per feature, stabilizing "
    "training."
)
bullet(
    "<b>Why only 2 layers?</b> &mdash; each GCN layer lets a node see one hop "
    "further. With 2 layers, each stop can see 2-hop neighbors. More layers cause "
    "<b>over-smoothing</b>: all node representations converge and become "
    "indistinguishable."
)
page_break()

# =====================================================================
# SLIDE 8 – MLP Baseline
# =====================================================================
slide_title("8. MLP Baseline")
body(
    "A <b>Multi-Layer Perceptron (MLP)</b> is a standard feedforward neural network. "
    "It takes each node's 12-dimensional feature vector and passes it through "
    "fully connected layers to produce a classification."
)
spacer()
subtitle("Architecture")
bullet("Layer 1: Linear(12 \u2192 64), BatchNorm, ReLU, Dropout")
bullet("Layer 2: Linear(64 \u2192 64), BatchNorm, ReLU, Dropout")
bullet("Layer 3: Linear(64 \u2192 2) &mdash; output logits")
spacer()
subtitle("Why Include It?")
body(
    "The MLP is a <b>graph-unaware baseline</b>. It classifies each stop using "
    "only that stop's own features &mdash; it never looks at edges or neighbors. "
    "Comparing GCN vs. MLP tells us whether the graph structure actually helps."
)
spacer()
subtitle("Interpretation")
bullet(
    "If GCN >> MLP: The graph structure carries signal; neighboring delays propagate."
)
bullet(
    "If MLP \u2248 GCN: The node features alone are sufficient; edges may be noisy."
)
bullet(
    "If MLP > GCN: The graph may be adding noise (e.g., random trip-based edges), "
    "or the GCN is over-smoothing."
)
spacer()
talking_point(
    "In the demo, MLP sometimes outperforms GCN because the synthetic edges are "
    "random trip-based connections. With real geographic or same-route edges, "
    "GCN would likely gain an advantage."
)
page_break()

# =====================================================================
# SLIDE 9 – Precision, Recall, F1, Support
# =====================================================================
slide_title("9. Precision, Recall, F1, and Support")
body(
    "These are the core classification metrics printed in the notebook's "
    "classification report. They are computed <b>per class</b> and then "
    "aggregated."
)
spacer()
subtitle("Definitions")
metric_table = [
    ["Metric", "Formula", "Intuition"],
    ["Precision", "TP / (TP + FP)", "Of all stops predicted at-risk, how many truly are?"],
    ["Recall", "TP / (TP + FN)", "Of all truly at-risk stops, how many did we catch?"],
    ["F1 Score", "2 \u00b7 P \u00b7 R / (P + R)", "Harmonic mean of precision & recall"],
    ["Support", "\u2014", "Number of true samples in that class"],
    ["F1-Macro", "mean(F1 per class)", "Equal weight to each class regardless of size"],
]
t = Table(metric_table, colWidths=[0.9*inch, 1.4*inch, 3.0*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a237e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), colors.white]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
spacer()
subtitle("Why Not Just Accuracy?")
body(
    "With imbalanced classes (e.g., 70% on-time, 30% at-risk), a model that predicts "
    "\"on-time\" for every stop achieves 70% accuracy while being completely useless "
    "for detecting at-risk stops. Precision and recall for the minority class reveal "
    "the true picture. <b>F1-Macro</b> gives equal weight to both classes, making it "
    "a fairer single number for imbalanced datasets."
)
spacer()
subtitle("Confusion Matrix")
body(
    "The confusion matrix is a 2\u00d72 table: rows = true labels, columns = predictions. "
    "The diagonal shows correct classifications (TN, TP); off-diagonal shows errors "
    "(FP, FN). The notebook plots side-by-side confusion matrices for GCN, MLP, "
    "and the Optuna-tuned model."
)
page_break()

# =====================================================================
# SLIDE 10 – ROC Curve
# =====================================================================
slide_title("10. ROC Curve and AUC")
body(
    "The <b>Receiver Operating Characteristic (ROC) curve</b> plots the trade-off "
    "between the True Positive Rate (Recall) and the False Positive Rate as you "
    "sweep the classification threshold from 0 to 1."
)
spacer()
subtitle("How to Read It")
bullet("<b>X-axis</b>: False Positive Rate = FP / (FP + TN)")
bullet("<b>Y-axis</b>: True Positive Rate = TP / (TP + FN) = Recall")
bullet(
    "The <b>diagonal dashed line</b> represents a random classifier (AUC = 0.5)."
)
bullet(
    "A perfect classifier hugs the <b>top-left corner</b> (AUC = 1.0)."
)
spacer()
subtitle("AUC (Area Under the Curve)")
body(
    "AUC summarizes the ROC curve as a single number between 0 and 1. It represents "
    "the probability that the model ranks a randomly chosen at-risk stop higher than "
    "a randomly chosen on-time stop. AUC > 0.8 is generally considered good."
)
spacer()
subtitle("In the Notebook")
body(
    "The notebook overlays ROC curves for all three models (GCN baseline, MLP baseline, "
    "and GCN Optuna) on one plot, making it easy to compare their discrimination ability."
)
spacer()
talking_point(
    "Consider: the curve that is furthest above the diagonal is the model with the best "
    "ability to separate the two classes across all possible decision thresholds."
)
page_break()

# =====================================================================
# SLIDE 11 – Training Curves & Error Metrics
# =====================================================================
slide_title("11. Training Curves &amp; Error Metrics")
body(
    "After each model finishes training, the notebook plots a multi-panel figure "
    "showing how metrics evolved <b>epoch by epoch</b>."
)
spacer()
subtitle("Panels")
bullet("<b>Loss</b> &mdash; cross-entropy loss. Train loss should decrease; val loss decreasing then rising signals overfitting.")
bullet("<b>Accuracy</b> &mdash; fraction of correctly classified nodes per epoch.")
bullet("<b>MAE</b> (Mean Absolute Error) &mdash; average absolute difference between predicted probability and true label.")
bullet("<b>MSE</b> (Mean Squared Error) &mdash; average squared difference; penalizes large errors more.")
bullet("<b>RMSE</b> (Root MSE) &mdash; square root of MSE, in the same units as the probability.")
spacer()
subtitle("Early Stopping")
body(
    "Training monitors validation loss. If it does not improve for <b>PATIENCE</b> "
    "(default 50) consecutive epochs, training halts and the best weights are "
    "restored. This prevents overfitting and saves time."
)
page_break()

# =====================================================================
# SLIDE 12 – Optuna
# =====================================================================
slide_title("12. How Optuna Hyperparameter Search Works")
body(
    "<b>Optuna</b> is an automated hyperparameter optimization framework. Instead of "
    "manually trying different values, Optuna intelligently searches the space."
)
spacer()
subtitle("The Search Loop")
body("For each of N_TRIALS (default 30):")
bullet("Optuna's <b>TPE sampler</b> (Tree-structured Parzen Estimator) proposes a new combination of hyperparameters.")
bullet("A GCN model is built with those hyperparameters and trained from scratch.")
bullet("The model is evaluated on the <b>validation set only</b> (never test) using <b>macro F1</b>.")
bullet("Optuna records the score and updates its model of which regions of the space are promising.")
spacer()
subtitle("What Gets Tuned")
hp_table = [
    ["Hyperparameter", "Search Range"],
    ["hidden_dim", "{16, 32, 64, 128}"],
    ["num_layers", "1 to 4"],
    ["dropout", "0.10 to 0.60 (step 0.05)"],
    ["learning_rate", "1e-4 to 1e-2 (log scale)"],
    ["weight_decay", "1e-5 to 1e-2 (log scale)"],
]
t = Table(hp_table, colWidths=[1.5*inch, 2.5*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a237e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), colors.white]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
spacer()
subtitle("TPE Sampler")
body(
    "Unlike grid search (try everything) or random search (try randomly), TPE builds "
    "a probabilistic model of past trials. It samples more from regions that produced "
    "good results. This is much more efficient &mdash; 30 trials with TPE can rival "
    "hundreds of random trials."
)
spacer()
subtitle("Why Macro F1 as the Objective?")
body(
    "With class imbalance, binary F1 can be zero if the minority class is never "
    "predicted. Macro F1 averages F1 across both classes, ensuring Optuna doesn't "
    "just optimize for the majority class."
)
page_break()

# =====================================================================
# SLIDE 13 – Feature / Hyperparameter Importance
# =====================================================================
slide_title("13. Hyperparameter Importance (fANOVA)")
body(
    "After the Optuna search, the notebook displays a <b>hyperparameter importance</b> "
    "bar chart. This answers: which hyperparameters mattered most for model "
    "performance?"
)
spacer()
subtitle("How It Works")
body(
    "Optuna uses <b>fANOVA</b> (functional Analysis of Variance). It fits a random "
    "forest on the (hyperparameter \u2192 score) data from all trials, then decomposes "
    "the variance in scores attributable to each hyperparameter. A hyperparameter that "
    "causes large swings in F1 gets high importance."
)
spacer()
subtitle("Interpreting the Chart")
bullet(
    "<b>High importance</b> (e.g., learning_rate): changing this parameter significantly "
    "affects performance. Spend time tuning it."
)
bullet(
    "<b>Low importance</b> (e.g., hidden_dim): performance is robust to changes. "
    "You can use a reasonable default."
)
spacer()
subtitle("Edge Case")
body(
    "If all trials produce identical scores (e.g., all F1 = 0.0), fANOVA cannot "
    "decompose zero variance and raises a RuntimeError. The notebook catches this "
    "and displays a fallback message."
)
spacer()
talking_point(
    "Highlight: this chart tells you where to focus your tuning effort in future "
    "experiments."
)
page_break()

# =====================================================================
# SLIDE 14 – Final Comparison Table
# =====================================================================
slide_title("14. Final Model Comparison")
body(
    "The last cell produces a summary table comparing GCN (baseline), MLP (baseline), "
    "and GCN (Optuna-tuned) across all metrics:"
)
spacer()
comp_table = [
    ["Category", "Metrics", "Goal"],
    ["Classification", "Accuracy, Precision, Recall, F1, F1-Macro", "Higher is better"],
    ["Discrimination", "ROC-AUC", "Higher is better"],
    ["Calibration / Error", "MAE, MSE, RMSE", "Lower is better"],
]
t = Table(comp_table, colWidths=[1.2*inch, 2.6*inch, 1.2*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a237e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), colors.white]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
spacer()
body(
    "Best values are highlighted: <b>bold/black</b> for the best in each column. "
    "Higher-is-better metrics highlight the max; lower-is-better highlight the min."
)
spacer()
subtitle("What to Discuss")
bullet("Did Optuna tuning improve the GCN over the default hyperparameters?")
bullet("Is the GCN leveraging graph structure, or is MLP competitive?")
bullet("Where is the model weakest? (Look at per-class recall for the minority class.)")
page_break()

# =====================================================================
# SLIDE 15 – Presentation Flow Cheat Sheet
# =====================================================================
slide_title("15. Suggested Presentation Flow")
body("Below is a recommended order for walking through the notebook live:")
spacer()
flow_table = [
    ["#", "Notebook Section", "Key Talking Points", "~Time"],
    ["1", "Title + Imports", "Introduce project, mention PyTorch Geometric", "1 min"],
    ["2", "Hyperparameters", "Show tuneable knobs; explain cfg dict", "2 min"],
    ["3", "Load Data", "Explain synthetic data; show node/edge counts", "2 min"],
    ["4", "Inspect Data", "Show tables; run delay histogram; explain threshold", "3 min"],
    ["5", "Stop Map (truth)", "Point out green/red dots; mention geographic layout", "2 min"],
    ["6", "Train GCN", "Run training; walk through epoch logs", "2 min"],
    ["7", "Training Curves", "Explain loss/acc/MAE panels; discuss convergence", "2 min"],
    ["8", "Train MLP", "Briefly train; compare speed", "1 min"],
    ["9", "Evaluate & Compare", "Classification reports; confusion matrices", "3 min"],
    ["10", "ROC Curves", "Explain ROC/AUC; compare models", "2 min"],
    ["11", "Predicted Map", "Compare vs ground truth map", "1 min"],
    ["12", "Optuna Search", "Explain TPE; run search; show importance", "3 min"],
    ["13", "Final Table", "Summarize all metrics; conclude", "2 min"],
]
t = Table(flow_table, colWidths=[0.3*inch, 1.3*inch, 2.7*inch, 0.6*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a237e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), colors.white]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t)
spacer()
body("<b>Total estimated time: ~26 minutes</b> (adjust by skipping or expanding sections)")
page_break()

# =====================================================================
# SLIDE 16 – Potential Questions & Answers
# =====================================================================
slide_title("16. Anticipated Questions")
spacer(0.05)

subtitle("Q: Why does MLP sometimes beat GCN?")
body(
    "The synthetic edges connect random trip-based stop pairs, which may not carry "
    "meaningful spatial signal. The GCN aggregates this noise. With real geographic "
    "or shared-route edges, GCN would likely gain an advantage."
)
spacer(0.1)

subtitle("Q: What would change with real MARTA data?")
body(
    "The codebase supports it. Set <font face='Courier'>use_mock: False</font> and "
    "provide GTFS URLs in the config. Expect more stops (~9,000), real route "
    "topology, and non-stationary delay patterns."
)
spacer(0.1)

subtitle("Q: How would you deploy this?")
body(
    "Periodically retrain on a rolling window of realtime data. Serve predictions via "
    "an API: given current conditions, flag stops with high delay risk. The trained "
    "model runs in milliseconds per inference."
)
spacer(0.1)

subtitle("Q: Why binary classification instead of regression?")
body(
    "Binary risk labels (on-time vs. at-risk) are more actionable for transit "
    "operators. The MAE/MSE/RMSE metrics bridge the gap by measuring how well "
    "the model's probability estimates match the true labels."
)

# =====================================================================
# Build PDF
# =====================================================================
doc.build(story)
print(f"PDF generated: {OUTPUT}")
