import pytorch_lightning as pl


class ShpDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name_or_path,
        max_seq_length,
        min_products_for_category,
        train_batch_size,
        val_batch_size,
        dataloader_num_workers,
        pin_memory,
        data_file_path=None,
        dataframe=None,
    ):
        super().__init__()
        self.data_file_path = data_file_path
        self.dataframe = dataframe
        self.min_products_for_category = min_products_for_category
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = pin_memory
        self.num_classes = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        self.train_dataset = mk_ds(
            list(self.df_trn.title), self.tokenizer, self.max_seq_length, self.ys_trn
        )
        self.eval_dataset = mk_ds(
            list(self.df_val.title), self.tokenizer, self.max_seq_length, self.ys_val
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
        )
