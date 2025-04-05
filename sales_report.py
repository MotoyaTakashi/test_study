import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# データベース接続設定
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///sales_report.db')
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# データベースモデル
class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    client = Column(String(100), nullable=False)
    description = Column(Text)
    created_at = Column(Date, default=datetime.now().date())

class SalesReport(Base):
    __tablename__ = 'sales_reports'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'))
    content = Column(Text, nullable=False)
    created_at = Column(Date, default=datetime.now().date())
    
    project = relationship("Project")

# データベースの作成
Base.metadata.create_all(engine)

def main():
    st.title("営業日報システム")
    
    # サイドバーで機能選択
    menu = st.sidebar.selectbox(
        "メニュー",
        ["日報入力", "案件管理", "日報一覧"]
    )
    
    if menu == "日報入力":
        show_report_input()
    elif menu == "案件管理":
        show_project_management()
    elif menu == "日報一覧":
        show_report_list()

def show_report_input():
    st.header("営業日報入力")
    
    # 日付選択
    report_date = st.date_input("日付", datetime.now())
    
    # 案件選択
    session = Session()
    projects = session.query(Project).all()
    project_options = {f"{p.name} ({p.client})": p.id for p in projects}
    
    if not project_options:
        st.warning("案件が登録されていません。先に案件を登録してください。")
        return
    
    selected_project = st.selectbox(
        "対応案件",
        options=list(project_options.keys())
    )
    
    # 日報内容入力
    content = st.text_area("日報内容", height=200)
    
    if st.button("保存"):
        if not content:
            st.error("日報内容を入力してください。")
            return
            
        try:
            new_report = SalesReport(
                date=report_date,
                project_id=project_options[selected_project],
                content=content
            )
            session.add(new_report)
            session.commit()
            st.success("日報を保存しました。")
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
        finally:
            session.close()

def show_project_management():
    st.header("案件管理")
    
    # 新規案件登録フォーム
    with st.form("new_project"):
        st.subheader("新規案件登録")
        project_name = st.text_input("案件名")
        client_name = st.text_input("顧客名")
        project_description = st.text_area("案件概要")
        
        if st.form_submit_button("登録"):
            if not project_name or not client_name:
                st.error("案件名と顧客名は必須です。")
            else:
                session = Session()
                try:
                    new_project = Project(
                        name=project_name,
                        client=client_name,
                        description=project_description
                    )
                    session.add(new_project)
                    session.commit()
                    st.success("案件を登録しました。")
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                finally:
                    session.close()
    
    # 案件一覧表示
    st.subheader("登録済み案件一覧")
    session = Session()
    projects = session.query(Project).all()
    
    if projects:
        df = pd.DataFrame([
            {
                "案件名": p.name,
                "顧客名": p.client,
                "概要": p.description,
                "登録日": p.created_at
            }
            for p in projects
        ])
        st.dataframe(df)
    else:
        st.info("登録されている案件はありません。")
    
    session.close()

def show_report_list():
    st.header("日報一覧")
    
    # 日付範囲選択
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("開始日", datetime.now())
    with col2:
        end_date = st.date_input("終了日", datetime.now())
    
    # 案件フィルター
    session = Session()
    projects = session.query(Project).all()
    project_options = ["すべて"] + [f"{p.name} ({p.client})" for p in projects]
    selected_project = st.selectbox("案件でフィルター", project_options)
    
    # 日報一覧表示
    query = session.query(SalesReport).join(Project)
    if selected_project != "すべて":
        project_id = projects[project_options.index(selected_project) - 1].id
        query = query.filter(SalesReport.project_id == project_id)
    
    reports = query.filter(
        SalesReport.date >= start_date,
        SalesReport.date <= end_date
    ).order_by(SalesReport.date.desc()).all()
    
    if reports:
        for report in reports:
            with st.expander(f"{report.date} - {report.project.name}"):
                st.write(f"顧客: {report.project.client}")
                st.write("内容:")
                st.write(report.content)
    else:
        st.info("該当する日報はありません。")
    
    session.close()

if __name__ == "__main__":
    main() 