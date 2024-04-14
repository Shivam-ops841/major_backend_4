import matplotlib.pyplot as plt

def create_bar_graph(categories, values, title="F1 Score", x_label="Categories", y_label="Values"):
    # Create a bar graph
    plt.stackplot(categories, values)

    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Display the graph
    plt.show()

# Example data
categories = ['KNN', 'CNN', 'Dtree', 'NB_Class','RF_Class','SVM']
values = [65, 55,58, 62, 56,54]

# Call the function to create the bar graph
create_bar_graph(categories, values, title="F1 Score", x_label="", y_label="")
