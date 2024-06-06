import streamlit as st
from pandas import DataFrame

import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
import re


st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
nltk.download('vader_lexicon')
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data= bytes_data.decode("utf-8")
    # st.text(data)
    df=preprocessor.preprocess(data)
    d = df.copy()

    # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Object
    sentiments = SentimentIntensityAnalyzer()

    # Creating different columns for (Positive/Negative/Neutral)
    d["po"] = [sentiments.polarity_scores(i)["pos"] for i in d["message"]]  # Positive
    d["ne"] = [sentiments.polarity_scores(i)["neg"] for i in d["message"]]  # Negative
    d["nu"] = [sentiments.polarity_scores(i)["neu"] for i in d["message"]]  # Neutral


    # To indentify true sentiment per row in message column
    def sentiment(d):
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0

        # Creating new column & Applying function


    d['value'] = d.apply(lambda row: sentiment(row), axis=1)

    # User names list
    user_list = d['user'].unique().tolist()

    # Sorting
    user_list.sort()

    # Insert "Overall" at index 0
    user_list.insert(0, "Overall")

    
    # st.dataframe(df)
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user= st.sidebar.selectbox("show analysis with respect to", user_list)
    if st.sidebar.button("show analysis"):
        num_messages,words,num_media_messages,num_links = helper.fetch_stats(selected_user, df)
        st.title("TOP STATISTICS")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total messages")
            st.title(num_messages)
        with col2:
            st.header("Total words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

            # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='red')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='purple')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

#     finding the busiest user
        if selected_user=='Overall':
            st.title('Most busy users')
            x,new_df=helper.most_busy_users(df)
            fig, ax= plt.subplots()
            col1,col2= st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='green')
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)


        st.title("wordcloud")
        df_wc= helper.create_wordcloud(selected_user,df)
        fig,ax= plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

    most_common_df = helper.most_common_words(selected_user, df)

    fig, ax = plt.subplots()

    ax.bar(most_common_df[0], most_common_df[1])
    plt.xticks(rotation='vertical')

    st.title('Most commmon words')
    st.pyplot(fig)

    # emoji analysis
    emoji_df = helper.emoji_helper(selected_user, df)
    st.title("Emoji Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(emoji_df)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(emoji_df[1].head(), autopct="%0.2f")
        st.pyplot(fig)

    # Percentage contributed

    if selected_user == 'Overall':
        st.title("Sentiment Analysis- most positive/neutral/negative users as a percent of their total messages")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text("Positive")
                # st.markdown("<h3 style='text-align: center; color: white;'>Most Positive Contribution</h3>",
                #             unsafe_allow_html=True)
            x = helper.percentage(d, 1)

                # Displaying
            st.dataframe(x)
        with col2:
            st.text("Neutral")
                # st.markdown("<h3 style='text-align: center; color: white;'>Most Neutral Contribution</h3>",
                #             unsafe_allow_html=True)
            y = helper.percentage(d, 0)

                # Displaying
            st.dataframe(y)
        with col3:
            st.text("Negative")
                # st.markdown("<h3 style='text-align: center; color: white;'>Most Negative Contribution</h3>",
                #             unsafe_allow_html=True)
            z = helper.percentage(d, -1)

                # Displaying
            st.dataframe(z)

        # Most Positive,Negative,Neutral User...
    if selected_user == 'Overall':
            # Getting names per sentiment
        x = (d['user'][d['value'] == 1].value_counts()/ (d['user'].value_counts()) *100 ).head(10)
        y = (d['user'][d['value'] == -1].value_counts()/ (d['user'].value_counts())*100).head(10)
        z = (d['user'][d['value'] == 0].value_counts()/ (d['user'].value_counts())*100).head(10)

        col1, col2, col3 = st.columns(3)
        with col1:
                # heading
            st.text('Most positive users-')

                # Displaying
            fig, ax = plt.subplots()
            ax.bar(x.index, x.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
                # heading
            st.text('Most Neutral users-')

                # Displaying
            fig, ax = plt.subplots()
            ax.bar(z.index, z.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
                # heading
            st.text('Most Negative users-')

                # Displaying
            fig, ax = plt.subplots()
            ax.bar(y.index, y.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
