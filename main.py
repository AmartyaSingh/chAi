from pdf_class import PDF_AI

if __name__ == "__main__":
    PDF_AI = PDF_AI()
    #file_path = PDF_AI.user_filepath_input()
    PDF_AI.create_chain()
    PDF_AI.loop_chat_with_file()