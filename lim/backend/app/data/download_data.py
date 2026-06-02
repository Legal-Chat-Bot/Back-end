import os
from datasets import load_dataset

BASE_PATH = '/content/drive/MyDrive/lim/backend/app/data'

def create_folders() :
    folders = [
        f'{BASE_PATH}/kr_legal_60k',
        f'{BASE_PATH}/aihub',
        f'{BASE_PATH}/merged'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print('폴더 생성 완료')

def download_huggingface():
    print('허깅페이스 데이터 로딩 중...')
    dataset = load_dataset("LDKSolutions/KR-legal-60K-dataset-jsonl")

    print(f'총 데이터 수: {len(dataset['train'])}건')
    print(f'컬럼: {dataset['train'].column_names}')

    save_path = f'{BASE_PATH}/kr_legal_60k'
    dataset.save_to_disk(save_path)
    print(f'저장 완료:{save_path}')
    return dataset

def check_sample(dataset):
    print('\n---샘플 데이터 확인---')
    sample = dataset['train'][0]
    for key, value in sample.items():
        print(f'\n{key}')
        print(str(value)[:200])

if __name__ == '__main__':
    create_folders()
    dataset=download_huggingface()
    check_sample(dataset)
    