import streamlit as st
import datetime
import pandas as pd

from gpt import gpt_answ
from infer import inference


# Initialize the session state for the diary DataFrame if it does not exist
if 'diary_df' not in st.session_state:
    st.session_state.diary_df = pd.DataFrame(columns=['Date', 'Text', 'Entry', 'Emoji'])

# Streamlit app title
st.title("나의 일기장")

# Diary entry form
with st.form("diary_form"):
    date = st.date_input("날짜", datetime.date.today())
    text = st.text_area("일기 내용을 입력하세요")
    submitted = st.form_submit_button("일기 저장")
    
    entry, emojis_df = None, None
    
    if submitted:
        # 사용자가 작업을 시작했을 때
        placeholder = st.empty()  # 장소 홀더 생성
        placeholder.info("일기에서 키워드 추출 중...")  # 초기 메시지 표시
        
        if not entry:
            entry = gpt_answ(text)
            # 필요한 경우 다른 메시지로 업데이트
            placeholder.success("키워드 추출 완료!")
        
        if not emojis_df:
            # 혹은 placeholder를 제거하려면
            placeholder.info("이모지로 변환 중...")
            emojis_df = inference(entry)
            placeholder.success("이모지로 변환 완료!")
        
        new_row_df = pd.DataFrame({'Date': [date], 'Text' : [text], 'Entry': [entry], 'Emoji' : [emojis_df['emoji'].tolist()]})
        # Concatenate with the existing DataFrame in session state
        st.session_state.diary_df = pd.concat([st.session_state.diary_df, new_row_df], ignore_index=True)
        st.success("일기가 저장되었습니다!")

# Display saved diary entries
if st.checkbox("저장된 일기 보기"):
    st.write(st.session_state.diary_df)
